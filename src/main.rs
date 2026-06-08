#![recursion_limit = "256"]

use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::Device,
};

use crate::model::Model;

mod data;
mod inference;
mod model;
mod training;

fn train_device() -> Device {
    return Device::flex();
}

fn inference_device() -> Device {
    return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
}

pub static ARTIFACT_DIR: &str = "/tmp/test-mnist";

fn main() {
    let skip_training = std::env::args().any(|arg| arg == "--skip-train");
    let only_training = std::env::args().any(|arg| arg == "--only-train");

    let train_device = train_device();
    if only_training {
        println!("Only training, skipping inference.");
        training::run(train_device.clone());
        println!("Training completed.");
        return;
    } else if skip_training {
        println!("Skipping training, using existing model in {ARTIFACT_DIR}");
    } else {
        training::run(train_device.clone());
        println!("Training completed.");
    }

    let inference_device = inference_device();

    let native = Model::new(&inference_device).load_record(
        CompactRecorder::new()
            .load(format!("{ARTIFACT_DIR}/model").into(), &inference_device)
            .expect("Trained model should exist; run train first"),
    );

    let variants = inference::quantize_variants(&native, inference::default_schemes());
    inference::compare_quantization(&native, &variants, &inference_device, 10000);
}
