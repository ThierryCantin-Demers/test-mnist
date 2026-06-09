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
    vec![
        QuantScheme {
            value: QuantValue::Q8S,
            level: QuantLevel::Tensor,
            ..base
        },
        QuantScheme {
            value: QuantValue::Q8S,
            level: QuantLevel::Block(BlockSize::new([32])),
            ..base
        },
        QuantScheme {
            value: QuantValue::Q4S,
            level: QuantLevel::Tensor,
            ..base
        },
        QuantScheme {
            value: QuantValue::Q4S,
            level: QuantLevel::Block(BlockSize::new([32])),
            ..base
        },
        QuantScheme {
            value: QuantValue::Q2S,
            level: QuantLevel::Tensor,
            ..base
        },
        QuantScheme {
            value: QuantValue::Q2S,
            level: QuantLevel::Block(BlockSize::new([32])),
            ..base
        },
    ]
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

/// Run `num_samples` test images through the full-precision model and every
/// quantized variant, then report each variant's accuracy and how closely it
/// agrees with the full-precision predictions.
pub fn compare_quantization(
    native: &Model,
    variants: &[(QuantScheme, Model)],
    device: &Device,
    num_samples: usize,
) {
    // Gather the first `num_samples` items from the test set into one batch.
    let dataset = MnistDataset::test();
    let items: Vec<_> = (0..num_samples).filter_map(|i| dataset.get(i)).collect();
    let n = items.len();
    let batch = MnistBatcher::default().batch(items, device);
    let images = batch.images;
    let targets = batch.targets;

    let native_pred = predictions(native, images.clone());
    let native_correct = count_equal(native_pred.clone(), targets.clone());

    let pct = |c: i64| 100.0 * c as f64 / n as f64;
    println!("\n=== Quantization comparison over {n} samples ===");
    println!(
        "{:<44} {:>9} {:>11} {:>14}",
        "Scheme", "Accuracy", "Agreement", "Disagreements"
    );
    println!(
        "{:<44} {:>8.2}% {:>11} {:>14}",
        "native (f32)",
        pct(native_correct),
        "—",
        "—"
    );

    for (scheme, model) in variants {
        let pred = predictions(model, images.clone());
        let correct = count_equal(pred.clone(), targets.clone());
        let agree = count_equal(pred, native_pred.clone());
        println!(
            "{:<44} {:>8.2}% {:>10.2}% {:>14}",
            scheme_label(scheme),
            pct(correct),
            pct(agree),
            n as i64 - agree,
        );
    }
    println!(
        "\nAgreement = predictions identical to the full-precision model \
        (higher means quantization changed fewer outputs)."
    );
}

/// Predicted class per image: argmax over the 10 logits, shape `[batch]`.
fn predictions(model: &Model, images: Tensor<3>) -> Tensor<1, Int> {
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
