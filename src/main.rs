#![recursion_limit = "256"]

use burn::{data::dataset::Dataset, tensor::Device};

mod data;
mod inference;
mod model;
mod training;

fn device() -> Device {
    // #[cfg(feature = "flex")]
    return Device::flex();

    // #[cfg(all(feature = "tch-gpu", not(target_os = "macos")))]
    // return Device::libtorch_cuda(burn::tensor::DeviceIndex::Default);

    // #[cfg(all(feature = "tch-gpu", target_os = "macos"))]
    // return Device::libtorch_mps();

    // #[cfg(feature = "tch-cpu")]
    // return Device::libtorch();

    // #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
    // return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);

    // #[cfg(feature = "cuda")]
    // return Device::cuda(burn::tensor::DeviceIndex::Default);

    // #[cfg(feature = "rocm")]
    // return Device::rocm(burn::tensor::DeviceIndex::Default);

    // #[cfg(feature = "remote")]
    // return Device::remote("ws://localhost:3000");

    // unreachable!("At least one backend will be selected.")
}

pub static ARTIFACT_DIR: &str = "/tmp/test-mnist";

fn main() {
    let skip_training = std::env::args().any(|arg| arg == "--skip-train");

    if skip_training {
        println!("Skipping training, using existing model in {ARTIFACT_DIR}");
    } else {
        training::run(device());
    }

    let infer_item = burn::data::dataset::vision::MnistDataset::test()
        .get(42)
        .unwrap();
    inference::infer(ARTIFACT_DIR, device(), infer_item.clone());
    inference::infer_quantized(ARTIFACT_DIR, device(), infer_item);
}
