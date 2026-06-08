use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{
        Device,
        quantization::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
    },
};

use crate::{data::MnistBatcher, model::Model};

pub fn infer(artifact_dir: &str, device: Device, item: MnistItem) {
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = Model::new(&device).load_record(record);

    let res = infer_inner(model, item, device);
    println!("Native: Predicted {} Expected {}", res.1, res.0);
}

pub fn infer_quantized(artifact_dir: &str, device: Device, item: MnistItem) {
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let scheme = QuantScheme {
        value: QuantValue::Q8S,
        param: QuantParam::F32,
        store: QuantStore::Native,
        level: QuantLevel::Tensor,
        mode: QuantMode::Symmetric,
    };
    let model = Model::new(&device).load_record(record).quantize(scheme);

    let res = infer_inner(model, item, device);
    println!("Quantized: Predicted {} Expected {}", res.1, res.0);
}

fn infer_inner(model: Model, item: MnistItem, device: Device) -> (u8, u8) {
    let label = item.label;
    let batcher = MnistBatcher::default();
    let batch = batcher.batch(vec![item], &device);
    let output = model.forward(batch.images);
    let predicted: u8 = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    (label, predicted)
}
