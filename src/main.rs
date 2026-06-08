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

#[allow(unreachable_code)]
fn device() -> Device {
    #[cfg(feature = "cpu")]
    return Device::flex();

    // #[cfg(all(feature = "tch-gpu", not(target_os = "macos")))]
    // return Device::libtorch_cuda(burn::tensor::DeviceIndex::Default);

    // #[cfg(all(feature = "tch-gpu", target_os = "macos"))]
    // return Device::libtorch_mps();

    // #[cfg(feature = "tch-cpu")]
    // return Device::libtorch();

    #[cfg(feature = "wgpu")]
    return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);

    // #[cfg(feature = "cuda")]
    // return Device::cuda(burn::tensor::DeviceIndex::Default);

    // #[cfg(feature = "rocm")]
    // return Device::rocm(burn::tensor::DeviceIndex::Default);

    // #[cfg(feature = "remote")]
    // return Device::remote("ws://localhost:3000");

    unreachable!("At least one backend will be selected.")
}

pub static ARTIFACT_DIR: &str = "/tmp/test-mnist";

fn main() {
    let skip_training = std::env::args().any(|arg| arg == "--skip-train");

    let device = device();
    if skip_training {
        println!("Skipping training, using existing model in {ARTIFACT_DIR}");
    } else {
        training::run(device.clone());
        println!("Training completed.")
    }

    let native = Model::new(&device).load_record(
        CompactRecorder::new()
            .load(format!("{ARTIFACT_DIR}/model").into(), &device)
            .expect("Trained model should exist; run train first"),
    );

    let variants = inference::quantize_variants(&native, inference::default_schemes());
    inference::compare_quantization(&native, &variants, &device, 1000);
}
