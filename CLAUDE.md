# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust workspace built on the [Burn](https://burn.dev) deep-learning framework. It trains an MLP on MNIST, then its real purpose: **comparing post-training quantization schemes** (Q8/Q4/Q2, per-tensor vs. block) against the full-precision model and reporting accuracy + agreement. This is an experimentation sandbox, not a product.

## Critical dependency: local Burn checkout

`Cargo.toml` points `burn` at `../burn` (a sibling git checkout, not crates.io):

```toml
burn = { path = "../burn/crates/burn", features = ["std", "tui", "train", "vision", "flex", "wgpu"], default-features = false }
```

The repo **will not build without `../burn` present**. This is deliberate — the code uses bleeding-edge Burn APIs (e.g. `QuantScheme`, `Quantizer`, `SupervisedTraining`, `Device::flex()`) that may not match released Burn. When an API doesn't compile, check the actual signatures in `../burn/crates/`, don't assume the published docs are current.

## Commands

```bash
cargo run                 # train, then run the quantization comparison
cargo run -- --only-train # train only, save model to /tmp/test-mnist
cargo run -- --skip-train # reuse saved model in /tmp/test-mnist, run comparison only
cargo build               # builds default member (test-mnist) only
cargo build --workspace   # also builds the benchmarks crate (currently non-compiling — see below)
```

Artifacts (checkpoints, `model`, `config.json`, training logs) go to `/tmp/test-mnist` (`ARTIFACT_DIR` in `main.rs`). The dir is wiped at the start of each training run.

## Architecture

Single binary crate `crates/test-mnist` (the default workspace member). Flow:

- **`main.rs`** — entry point and device selection. `train_device()` / `inference_device()` both currently return WGPU. Dispatches between train-only, skip-train, and the default train-then-compare path.
- **`model.rs`** — `Model`: a 4-layer MLP (784→4096→4096→4096→10, GELU, dropout). Implements Burn's `TrainStep`/`InferenceStep`. `Model::quantize(scheme)` applies a `Quantizer` (MinMax calibration) to each layer's weights.
- **`data.rs`** — `MnistBatcher` turns `MnistItem`s into batched tensors, normalizing with the standard PyTorch MNIST mean/std (0.1307 / 0.3081).
- **`training.rs`** — `run()`: AdamW + composed (warmup → cosine → linear-decay) LR schedule, 5 epochs, batch 256. Splits the 60k train set into 55k train / 5k validation; evaluates on the test set; saves model + config.
- **`inference.rs`** — the quantization study. `default_schemes()` defines the schemes to compare, `quantize_variants()` produces one model per scheme, `compare_quantization()` prints an accuracy/agreement table vs. the f32 baseline.

## Determinism

Training only reproduces bit-for-bit with `num_workers = 1` (the config default). Multi-worker dataloading makes batch order nondeterministic even with a fixed seed. Don't raise `num_workers` in `training.rs` if reproducibility matters. (The test dataloader uses 2 workers because it doesn't affect trained weights.)

## Quantization scheme constraints

When adding schemes to `inference.rs::default_schemes()`, note the model's final layer outputs a 10-wide dim. This breaks:
- **`Block` schemes** that tile the last dim when it isn't a multiple of the block size.
- **Sub-byte values** (`Q4*`, etc.) that pack 4 values per `u32` and need the last dim divisible by 4.

The header comment on `default_schemes()` documents this — read it before extending the list.

## benchmarks crate

`benchmarks/` is **work-in-progress and does not compile** — `benches/dequantize.rs` references undefined fields and the `burnbench` dependency is commented out in its `Cargo.toml`. It's excluded from `default-members`, so plain `cargo build`/`cargo run` skip it. Don't treat its current state as a regression.
