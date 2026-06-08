use std::sync::Arc;

use burn::{
    config::Config,
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{transform::PartialDataset, vision::MnistDataset},
    },
    lr_scheduler::{
        composed::ComposedLrSchedulerConfig, cosine::CosineAnnealingLrSchedulerConfig,
        linear::LinearLrSchedulerConfig,
    },
    module::Module,
    optim::AdamWConfig,
    record::CompactRecorder,
    tensor::Device,
    train::{
        EvaluatorBuilder, Learner, SupervisedTraining,
        metric::{AccuracyMetric, LearningRateMetric, LossMetric},
    },
};

use crate::{ARTIFACT_DIR, data::MnistBatcher, model::Model};

#[derive(Config, Debug)]
pub struct MnistTrainingConfig {
    #[config(default = 5)]
    pub num_epochs: usize,

    #[config(default = 256)]
    pub batch_size: usize,

    #[config(default = 8)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamWConfig,
}

pub fn run(device: Device) {
    create_artifact_dir(ARTIFACT_DIR);

    // Config
    let config_optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(true)
        .with_weight_decay(5e-5);

    let config = MnistTrainingConfig::new(config_optimizer);

    device.seed(config.seed);
    let autodiff_device = device.clone().autodiff();

    let model = Model::new(&autodiff_device);

    let dataset_train_original = Arc::new(MnistDataset::train());
    let dataset_train = PartialDataset::new(dataset_train_original.clone(), 0, 55_000);
    let dataset_valid = PartialDataset::new(dataset_train_original.clone(), 55_000, 60_000);

    let batcher = MnistBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_valid);
    let lr_scheduler = ComposedLrSchedulerConfig::new()
        .cosine(CosineAnnealingLrSchedulerConfig::new(1.0, 2000))
        // Warmup
        .linear(LinearLrSchedulerConfig::new(1e-8, 1.0, 2000))
        .linear(LinearLrSchedulerConfig::new(1e-2, 1e-6, 10000));

    let training = SupervisedTraining::new(ARTIFACT_DIR, dataloader_train, dataloader_valid)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        lr_scheduler.init().unwrap(),
    ));

    let model = result.model.clone();

    let dataloader_test = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(config.batch_size)
        .num_workers(2)
        .build(MnistDataset::test());

    // let renderer = EvaluatorBuilder::new(ARTIFACT_DIR)
    EvaluatorBuilder::new(ARTIFACT_DIR)
        .renderer(result.renderer)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .summary()
        .build(model.clone())
        .eval("test", dataloader_test.clone());

    model
        .save_file(format!("{ARTIFACT_DIR}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}
