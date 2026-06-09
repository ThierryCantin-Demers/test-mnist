use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, vision::MnistDataset},
    },
    tensor::{
        Device, Int, Tensor,
        quantization::{
            BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
        },
    },
};

use crate::{data::MnistBatcher, model::Model};

/// A set of quantization schemes to compare against the full-precision model.
///
/// These are all per-tensor Q8 variants because this model's final layer has a
/// 10-wide output dimension: `Block` schemes (which tile the last dim) and
/// sub-byte values like `Q4*` (which require packing 4 values per u32, needing
/// the last dim to be a multiple of 4) both fail on `fc3`. Add more here once
/// you account for that.
pub fn default_schemes() -> Vec<QuantScheme> {
    let base = QuantScheme {
        value: QuantValue::Q8S,
        param: QuantParam::F32,
        store: QuantStore::PackedU32(0),
        level: QuantLevel::Tensor,
        mode: QuantMode::Symmetric,
    };

    let mut cases = vec![];
    for value in [QuantValue::Q8S, QuantValue::Q4S, QuantValue::Q2S] {
        for level in [
            QuantLevel::Tensor,
            QuantLevel::Block(BlockSize::new([16, 16])),
            QuantLevel::Block(BlockSize::new([128])),
            QuantLevel::Block(BlockSize::new([32])),
            QuantLevel::Block(BlockSize::new([16])),
        ] {
            cases.push(QuantScheme {
                value,
                level,
                ..base
            });
        }
    }

    cases
}

/// Quantize the full-precision model once per scheme, keeping each scheme paired
/// with its resulting model so results can be labeled. Each variant gets its own
/// clone of the weights (quantization consumes the model).
pub fn quantize_variants(native: &Model, schemes: Vec<QuantScheme>) -> Vec<(QuantScheme, Model)> {
    schemes
        .into_iter()
        .map(|scheme| (scheme, native.clone().quantize(scheme)))
        .collect()
}

/// A fixed test batch plus the full-precision baseline predictions, computed
/// once and reused to score every model so accuracy and agreement are measured
/// over the exact same inputs.
pub struct Eval {
    pub images: Tensor<3>,
    pub targets: Tensor<1, Int>,
    pub native_pred: Tensor<1, Int>,
    pub n: usize,
}

/// Build the shared evaluation batch from the first `num_samples` test images
/// and record the full-precision model's predictions as the baseline.
pub fn prepare_eval(native: &Model, device: &Device, num_samples: usize) -> Eval {
    let dataset = MnistDataset::test();
    let items: Vec<_> = (0..num_samples).filter_map(|i| dataset.get(i)).collect();
    let n = items.len();
    let batch = MnistBatcher::default().batch(items, device);
    let native_pred = predictions(native, batch.images.clone());
    Eval {
        images: batch.images,
        targets: batch.targets,
        native_pred,
        n,
    }
}

/// One model's quality relative to the full-precision baseline.
#[derive(Clone, Copy, Debug)]
pub struct Quality {
    /// Percent of samples classified correctly.
    pub accuracy: f64,
    /// Percent of samples whose prediction is identical to full precision.
    pub agreement: f64,
    /// Number of samples whose prediction differs from full precision.
    pub disagreements: i64,
}

/// Score a model's already-computed `pred` against the shared [`Eval`] baseline.
///
/// Takes predictions rather than the model so callers that have already run the
/// forward pass (e.g. the timing benchmark) don't pay for a second one.
pub fn quality(pred: &Tensor<1, Int>, eval: &Eval) -> Quality {
    let correct = count_equal(pred.clone(), eval.targets.clone());
    let agree = count_equal(pred.clone(), eval.native_pred.clone());
    let pct = |c: i64| 100.0 * c as f64 / eval.n as f64;
    Quality {
        accuracy: pct(correct),
        agreement: pct(agree),
        disagreements: eval.n as i64 - agree,
    }
}

/// Run `num_samples` test images through the full-precision model and every
/// quantized variant, then report each variant's accuracy and how closely it
/// agrees with the full-precision predictions.
pub fn compare_quantization(
    native: &Model,
    variants: &[(QuantScheme, Model)],
    device: &Device,
    num_samples: usize,
) {
    let eval = prepare_eval(native, device, num_samples);
    let native_quality = quality(&eval.native_pred, &eval);

    println!("\n=== Quantization comparison over {} samples ===", eval.n);
    println!(
        "{:<44} {:>9} {:>11} {:>14}",
        "Scheme", "Accuracy", "Agreement", "Disagreements"
    );
    println!(
        "{:<44} {:>8.2}% {:>11} {:>14}",
        "native (f32)", native_quality.accuracy, "—", "—"
    );

    for (scheme, model) in variants {
        let q = quality(&predictions(model, eval.images.clone()), &eval);
        println!(
            "{:<44} {:>8.2}% {:>10.2}% {:>14}",
            scheme_label(scheme),
            q.accuracy,
            q.agreement,
            q.disagreements,
        );
    }
    println!(
        "\nAgreement = predictions identical to the full-precision model \
        (higher means quantization changed fewer outputs)."
    );
}

/// Predicted class per image: argmax over the 10 logits, shape `[batch]`.
pub fn predictions(model: &Model, images: Tensor<3>) -> Tensor<1, Int> {
    model.forward(images).argmax(1).flatten::<1>(0, 1)
}

/// Number of positions where the two label tensors are equal.
fn count_equal(a: Tensor<1, Int>, b: Tensor<1, Int>) -> i64 {
    a.equal(b).int().sum().into_scalar::<i64>()
}

/// Compact, human-readable description of a scheme for the results table.
fn scheme_label(scheme: &QuantScheme) -> String {
    let level = match &scheme.level {
        QuantLevel::Tensor => "per-tensor".to_string(),
        QuantLevel::Block(block) => format!("block({})", block.num_elements()),
    };
    format!(
        "{:?} {} {:?} {:?}",
        scheme.value, level, scheme.mode, scheme.store
    )
}
