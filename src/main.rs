mod model;
mod synth;
mod tokenizer;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss::cross_entropy, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use model::{Config, GptLog};
use tokenizer::CharTokenizer;

// sample a batch: windows of `block` chars, target shifted by +1
fn sample_batch(
    data: &[u32],
    batch: usize,
    block: usize,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut xs = Vec::with_capacity(batch * block);
    let mut ys = Vec::with_capacity(batch * block);
    for _ in 0..batch {
        let i = fastrand::usize(0..(data.len() - block - 1));
        xs.extend_from_slice(&data[i..i + block]);
        ys.extend_from_slice(&data[i + 1..i + block + 1]);
    }
    let x = Tensor::from_vec(xs, (batch, block), dev)?;
    let y = Tensor::from_vec(ys, (batch, block), dev)?;
    Ok((x, y))
}

// Deterministic weight init. candle's CPU backend refuses to be seeded
// (Device::set_seed bails on Cpu), so instead of relying on its internal RNG we
// overwrite every freshly created variable with values drawn from our own
// seeded generator. Variables are visited in sorted-name order so the RNG is
// consumed in a fixed sequence: same seed => bit-identical run.
fn init_deterministic(varmap: &VarMap, seed: u64) -> Result<()> {
    fastrand::seed(seed);
    let dev = Device::Cpu;
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for name in names {
        let var = &data[name];
        let dims = var.dims().to_vec();
        let n: usize = dims.iter().product();
        let vals: Vec<f32> = if name.ends_with("bias") {
            vec![0f32; n]
        } else if name.contains("ln") && name.ends_with("weight") {
            vec![1f32; n]
        } else {
            // uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)), as candle's linear init
            let b = 1.0 / (*dims.last().unwrap() as f64).sqrt();
            (0..n)
                .map(|_| ((fastrand::f64() * 2.0 - 1.0) * b) as f32)
                .collect()
        };
        var.set(&Tensor::from_vec(vals, dims, &dev)?)?;
    }
    Ok(())
}

fn train(corpus_ids: &[u32], cfg: &Config, steps: usize, seed: u64) -> Result<(VarMap, GptLog)> {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = GptLog::new(cfg, vb)?;
    init_deterministic(&varmap, seed)?;

    let params = ParamsAdamW {
        lr: 3e-3,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    let (batch, block) = (32usize, cfg.block_size);

    for step in 0..steps {
        let (x, y) = sample_batch(corpus_ids, batch, block, &dev)?;
        let logits = model.forward(&x)?;
        let (b, t, v) = logits.dims3()?;
        let loss = cross_entropy(&logits.reshape((b * t, v))?, &y.reshape(b * t)?)?;
        opt.backward_step(&loss)?;
        if step % 500 == 0 || step == steps - 1 {
            println!("  step {step:5}  loss {:.4}", loss.to_scalar::<f32>()?);
        }
    }
    // rebuild the model from the trained weights in the varmap (to return owned)
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = GptLog::new(cfg, vb)?;
    Ok((varmap, model))
}

// perplexity of a single line under the model
fn line_perplexity(model: &GptLog, tok: &CharTokenizer, line: &str) -> Result<f32> {
    let ids = tok.encode(line);
    if ids.len() < 2 {
        return Ok(1.0);
    }
    let n = ids.len().min(model.block_size + 1);
    let dev = Device::Cpu;
    let x = Tensor::from_vec(ids[..n - 1].to_vec(), (1, n - 1), &dev)?;
    let logits = model.forward(&x)?;
    let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
    let logp = logp.squeeze(0)?; // [n-1, vocab]
    let mut nll = 0f32;
    for pos in 0..(n - 1) {
        let target = ids[pos + 1] as usize;
        let lp: f32 = logp.get(pos)?.get(target)?.to_scalar()?;
        nll -= lp;
    }
    Ok((nll / (n - 1) as f32).exp())
}

// Anomalies in the Apache DOMAIN: they start from real lines and corrupt them
// the way a real system would under failure or attack.
fn apache_anomaly(kind: &str, real_lines: &[&str]) -> String {
    let base = real_lines[fastrand::usize(0..real_lines.len())];
    match kind {
        // syntactic: garbage/binary bytes in the middle of the log (corruption)
        "garbage" => "[Sun Dec 04 04:47:44 2005] [error] ####@@@!!!~~~%%%^^^&&&***(((|||}}}".to_string(),
        // syntactic: injection - a payload that never shows up in a normal log
        "injection" => format!(
            "[Sun Dec 04 04:47:44 2005] [error] 1' OR '1'='1 /../../private/config"
        ),
        // syntactic: a made-up level that does not exist in the Apache vocabulary
        "bad_level" => base.replacen("[notice]", "[CATASTROPHE]", 1).replacen("[error]", "[CATASTROPHE]", 1),
        // semantic: an anomalous magnitude in a numeric token (loop/flood)
        "flood" => "[Sun Dec 04 04:47:44 2005] [notice] jk2_init() Found child 99999999999999 in scoreboard slot 99999".to_string(),
        _ => base.to_string(),
    }
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}
fn std(v: &[f32], m: f32) -> f32 {
    (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
}

// One seed drives everything: weight init, batch sampling, synthetic lines and
// the anomalies. Change it and you get a different, equally reproducible run.
const SEED: u64 = 7;

fn main() -> Result<()> {
    fastrand::seed(SEED);
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("synth");

    // ---- training corpus ----
    let corpus = if mode == "real" {
        // REAL data: Apache error log (Loghub). Trains on the "normal" only.
        let raw = std::fs::read_to_string("data/Apache_2k.log")
            .expect("download data/Apache_2k.log from Loghub");
        // use 90% for training
        let lines: Vec<&str> = raw.lines().collect();
        let cut = lines.len() * 9 / 10;
        lines[..cut].join("\n") + "\n"
    } else {
        synth::build_corpus(4000)
    };

    let tok = CharTokenizer::from_corpus(&corpus);
    let ids = tok.encode(&corpus);
    println!(
        "mode={}  corpus={} chars  vocab={}  lines={}",
        mode,
        corpus.len(),
        tok.vocab_size(),
        corpus.lines().count()
    );

    let cfg = Config {
        vocab_size: tok.vocab_size(),
        block_size: 48,
        n_embd: 48,
        n_head: 4,
        n_layer: 1,
    };

    println!("training (CPU)...");
    let steps: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1500);
    let t0 = std::time::Instant::now();
    let (varmap, model) = train(&ids, &cfg, steps, SEED)?;
    let nparams: usize = varmap
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();
    println!(
        "params={}  time={:.1}s",
        nparams,
        t0.elapsed().as_secs_f32()
    );

    // ---- evaluation: perplexity of normal lines vs anomalies (per domain) ----
    let (npp, kinds): (Vec<f32>, Vec<&str>) = if mode == "real" {
        // normal baseline = the 10% held-out real lines the model did NOT see
        let raw = std::fs::read_to_string("data/Apache_2k.log").unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        let cut = lines.len() * 9 / 10;
        let held: Vec<&str> = lines[cut..].to_vec();
        let npp: Vec<f32> = held
            .iter()
            .map(|l| line_perplexity(&model, &tok, l).unwrap_or(1.0))
            .collect();
        (npp, vec!["garbage", "injection", "bad_level", "flood"])
    } else {
        let normals: Vec<String> = (0..200).map(|_| synth::normal_line()).collect();
        let npp: Vec<f32> = normals
            .iter()
            .map(|l| line_perplexity(&model, &tok, l).unwrap_or(1.0))
            .collect();
        (npp, vec!["garbage", "weird_method", "huge_lat", "bad_code"])
    };

    let m = mean(&npp);
    let sd = std(&npp, m);
    let thr = m + 2.0 * sd;
    println!("\n=== perplexity ===");
    println!("normal: mean {:.2}  std {:.2}  threshold(mean+2sd) {:.2}", m, sd, thr);

    // real lines used as the base for the anomalies in real mode
    let raw = std::fs::read_to_string("data/Apache_2k.log").unwrap_or_default();
    let real_lines: Vec<&str> = raw.lines().collect();

    for kind in kinds {
        let pp: Vec<f32> = (0..100)
            .map(|_| {
                let line = if mode == "real" {
                    apache_anomaly(kind, &real_lines)
                } else {
                    synth::anomaly(kind)
                };
                line_perplexity(&model, &tok, &line).unwrap_or(1.0)
            })
            .collect();
        let mp = mean(&pp);
        let det = pp.iter().filter(|&&p| p > thr).count();
        println!(
            "  {:12}  pp {:7.2}  ({:5.1}x)  detection {}%",
            kind,
            mp,
            mp / m,
            det
        );
    }
    Ok(())
}
