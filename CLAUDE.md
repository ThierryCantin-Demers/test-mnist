# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust workspace built on the [Burn](https://burn.dev) deep-learning framework. It trains an MLP on MNIST, then its real purpose: **comparing post-training quantization schemes** (Q8/Q4/Q2, per-tensor vs. block) against the full-precision model and reporting accuracy + agreement, and **benchmarking the dequantization cost** of each scheme. This is an experimentation sandbox, not a product.

## Critical dependency: local Burn + cubecl checkouts

The workspace `Cargo.toml` pulls `burn` from a sibling git checkout (not crates.io) and patches both `burn` and `cubecl` to sibling checkouts:

```toml
[workspace.dependencies]
burn = {
    path = "../burn/crates/burn",
    features = ["std", "tui", "train", "vision", "flex", "wgpu", "cpu"],
    default-features = false,
}

[patch."https://github.com/tracel-ai/burn"]
burn = { path = "../burn/crates/burn" }

[patch."https://github.com/tracel-ai/cubecl"]
cubecl = { path = "../cubecl/crates/cubecl" }
cubecl-common = { path = "../cubecl/crates/cubecl-common" }
```

The repo **will not build without `../burn` and `../cubecl` present** as sibling checkouts. This is deliberate — the code uses bleeding-edge Burn APIs (e.g. `QuantScheme`, `Quantizer`, `SupervisedTraining`, `Device::flex()`) that may not match released Burn. The cubecl patch is what makes the `burnbench` git dependency in the benchmarks crate resolve against the same local cubecl. When an API doesn't compile, check the actual signatures in `../burn/crates/` and `../cubecl/crates/`, don't assume the published docs are current.

## Commands

`crates/test-mnist` is both a library and **two binaries** — there is no single `main.rs` or CLI-flag dispatch. Train and infer are separate binaries:

```bash
cargo run --bin train          # train, save model + config to /tmp/test-mnist
cargo run --bin infer          # reuse saved model, run the quantization comparison
cargo build                    # builds the default member (test-mnist lib + both bins)
cargo build --workspace        # also builds the benchmarks crate
```

`cargo run` with no `--bin` is ambiguous (two binaries) and will error — always pass `--bin train` or `--bin infer`. `infer` expects a trained model in `/tmp/test-mnist`; run `train` first.

Artifacts (checkpoints, `model`, `config.json`, training logs) go to `/tmp/test-mnist` (`ARTIFACT_DIR`, defined in `lib.rs`). The dir is wiped at the start of each training run.

## Architecture

Workspace with two crates: `crates/test-mnist` (the default member) and `crates/benchmarks` (a member, excluded from `default-members`).

`crates/test-mnist` is a library (`lib.rs`) plus two thin binaries under `bin/`:

- **`lib.rs`** — crate root. Re-exports the modules, defines `ARTIFACT_DIR`, and holds device selection: `train_device()` / `inference_device()` both currently return WGPU (`Device::wgpu(DefaultDevice)`).
- **`bin/train.rs`** — `train` binary. Calls `training::run(train_device())`.
- **`bin/infer.rs`** — `infer` binary. Loads the saved model, then runs `inference::quantize_variants` + `inference::compare_quantization`.
- **`model.rs`** — `Model`: a 4-layer MLP (784→4096→4096→4096→10, GELU, dropout; `NUM_LAYERS = 4`, `LAYER_SIZE = 4096`). Implements Burn's `TrainStep`/`InferenceStep`. `Model::quantize(scheme)` applies a `Quantizer` (MinMax calibration) to the **hidden** layers only — it pops the final 10-wide output layer, quantizes the rest, and pushes the output layer back in full precision. `Model::quantized_weights()` returns just those quantized hidden-layer weight tensors (used by the benchmarks).
- **`data.rs`** — `MnistBatcher` turns `MnistItem`s into batched tensors, normalizing with the standard PyTorch MNIST mean/std (0.1307 / 0.3081).
- **`training.rs`** — `run()`: AdamW (cautious, weight decay 5e-5) + composed (warmup → cosine → linear-decay) LR schedule. `MnistTrainingConfig` defaults: 5 epochs, batch 256, `num_workers = 1`, seed 42. Splits the 60k train set into 55k train / 5k validation; evaluates on the test set; saves model + config.
- **`inference.rs`** — the quantization study. `default_schemes()` builds the schemes to compare; `quantize_variants()` produces one model per scheme; `compare_quantization()` prints an accuracy/agreement table vs. the f32 baseline. Also exposes the reusable pieces the benchmarks share: `prepare_eval()` → `Eval`, `predictions()`, and `quality()` → `Quality` (accuracy, agreement, disagreement count vs. full precision).

## Determinism

Training only reproduces bit-for-bit with `num_workers = 1` (the config default). Multi-worker dataloading makes batch order nondeterministic even with a fixed seed. Don't raise `num_workers` in `training.rs` if reproducibility matters. (The test dataloader uses 2 workers because it doesn't affect trained weights.)

## Quantization scheme constraints

`default_schemes()` builds the cross-product of three values (`Q8S`, `Q4S`, `Q2S`) × five levels (per-`Tensor`, plus `Block` sizes `[16,16]`, `[128]`, `[32]`, `[16]`).

This cross-product works **because `Model::quantize` excludes the 10-wide output layer** (see `model.rs`). The 10-wide final dim is what would otherwise break:
- **`Block` schemes** that tile the last dim when it isn't a multiple of the block size.
- **Sub-byte values** (`Q4*`, etc.) that pack 4 values per `u32` and need the last dim divisible by 4.

The quantized hidden layers (784→4096→4096→4096) all have dims that satisfy these, so every scheme in the cross-product is valid. If you ever start quantizing the output layer, re-check these constraints. Note: the doc comment on `default_schemes()` itself still claims "all per-tensor Q8" — that comment is stale relative to the code; trust the code.

## benchmarks crate

`crates/benchmarks` times the cost of dequantization per scheme using `burnbench` (a git dependency, resolved via the workspace cubecl patch). It depends on `test-mnist` and reuses `default_schemes`, `prepare_eval`, `predictions`, `quality`, and `Model::quantized_weights`. It loads the trained model from `/tmp/test-mnist`, so **run `cargo run --bin train` first**.

Two benches (both `harness = false`):
- **`benches/dequantize.rs`** — end-to-end inference timing: full forward pass per scheme on a small (256-image) batch, the weight-bandwidth-bound regime where dequant-on-read cost shows up. Reports timing alongside accuracy/agreement.
- **`benches/dequantize_kernel.rs`** — the dequantize kernel in isolation: times `Tensor::dequantize` on the model's quantized weights, no matmul.

Both share a `BenchBackend` enum (Wgpu / Flex / Cpu — Wgpu by default), warm up the device before timing, and report the best of `RUNS` runs (smallest `min`) to suppress integrated-GPU clock noise. `src/bin/burnbench.rs` is the `burnbench` runner entry point.

The crate is excluded from `default-members`, so plain `cargo build`/`cargo run` skip it; build it explicitly with `cargo build --workspace` or `cargo build -p benchmarks`.
