use burn::tensor::Device;

pub mod data;
pub mod inference;
pub mod model;
pub mod training;

pub static ARTIFACT_DIR: &str = "/tmp/test-mnist";

pub fn train_device() -> Device {
    Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)
}

pub fn inference_device() -> Device {
    Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)
}
