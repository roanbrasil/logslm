# logslm — a log-specialist SLM, in Rust

A tiny char-level GPT (~35k parameters) that learns the "grammar" of a service's
logs and detects anomalies by perplexity, running 100% on CPU. No cloud, no GPU.

Trained only on normal lines: the model never sees a labelled anomaly. Whatever
surprises it — a perplexity above `mean + 2sd` of the normal baseline — is flagged.

## Running

```bash
# didactic mode (synthetic log generator)
cargo run --release -- synth 2500

# real mode (Apache error log from Loghub, in data/Apache_2k.log)
cargo run --release -- real 3000
```

The first argument is the mode (`synth`|`real`), the second is the number of
training steps. Run from the project root — the data path is relative.

## Reproducibility

Runs are bit-identical: same seed, same output, down to every loss and every
perplexity. A single `SEED` in `src/main.rs` drives weight init, batch sampling,
the synthetic lines and the anomalies.

candle's CPU backend cannot be seeded — `Device::set_seed` bails on `Cpu` — so
the weights are not left to its internal RNG: `init_deterministic` overwrites
every variable with values drawn from our own seeded generator, visiting them in
sorted-name order so the draw sequence is fixed. Determinism holds across thread
counts too (verified with `RAYON_NUM_THREADS=1` against the default). Only the
reported wall-clock time varies between runs.

The reference output for the two commands above:

```
mode=synth  corpus=160230 chars  vocab=49  lines=4000
  step     0  loss 3.9964
  step  2499  loss 1.4649
params=35233

=== perplexity ===
normal: mean 4.55  std 1.32  threshold(mean+2sd) 7.18
  garbage       pp  512.89  (112.8x)  detection 100%
  weird_method  pp   29.07  (  6.4x)  detection 100%
  huge_lat      pp    5.05  (  1.1x)  detection 0%
  bad_code      pp    4.61  (  1.0x)  detection 0%
```

```
mode=real  corpus=152331 chars  vocab=51  lines=1800
  step     0  loss 4.1293
  step  2999  loss 1.9951
params=35427

=== perplexity ===
normal: mean 7.56  std 1.84  threshold(mean+2sd) 11.23
  garbage       pp   27.81  (  3.7x)  detection 100%
  injection     pp   29.06  (  3.8x)  detection 100%
  bad_level     pp   38.82  (  5.1x)  detection 100%
  flood         pp    6.05  (  0.8x)  detection 0%
```

Training starts at a loss of about `ln(vocab)` (3.89 for 49 characters, 3.93 for
51) — the loss of a model guessing uniformly, which is the check that the
initialization is sane: the model really does start out knowing nothing.

## What it catches, what it misses

Syntactic anomalies (garbage bytes, invalid method, SQL injection, nonexistent
log level) jump far above the normal baseline and are detected 100% of the time.
Semantic anomalies (absurd latency, contradictory status code, numeric flood) sit
near the normal baseline and pass unnoticed: a character-level model has no notion
of magnitude. That boundary is a property of the architecture — for those cases,
a one-line threshold rule beats any model.

## Real data

`data/Apache_2k.log` comes from Loghub (logpai/loghub), a 2k-line sample of the
Apache HTTP Server error log. Fetch it with:

```bash
curl -sSL https://raw.githubusercontent.com/logpai/loghub/master/Apache/Apache_2k.log \
  -o data/Apache_2k.log
```

## Layout

- `src/tokenizer.rs` — deterministic char-level tokenizer
- `src/model.rs`     — decoder-only GPT in candle (causal attention, block, model)
- `src/synth.rs`     — didactic log generator + anomalies
- `src/main.rs`      — training, perplexity, evaluation

Cite Loghub: Jieming Zhu et al., "Loghub: A Large Collection of System Log
Datasets for AI-driven Log Analytics", ISSRE 2023.
