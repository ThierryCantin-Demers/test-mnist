#![recursion_limit = "256"]

use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
};
use test_mnist::{ARTIFACT_DIR, inference, inference_device, model::Model};

fn main() {
    let inference_device = inference_device();

    let native = Model::new(&inference_device).load_record(
        CompactRecorder::new()
            .load(format!("{ARTIFACT_DIR}/model").into(), &inference_device)
            .expect("Trained model should exist; run train first"),
    );

    let variants = inference::quantize_variants(&native, inference::default_schemes());
    inference::compare_quantization(&native, &variants, &inference_device, 10000);
}
