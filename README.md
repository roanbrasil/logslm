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
