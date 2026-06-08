use burn::{
    Tensor,
    module::{Module, Quantizer},
    nn::{self, loss::CrossEntropyLossConfig},
    tensor::{
        Device,
        quantization::{Calibration, QuantScheme},
    },
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::data::MnistBatch;

#[derive(Module, Debug)]
pub struct Model {
    dropout: nn::Dropout,
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
    activation: nn::Gelu,
}

const NUM_CLASSES: usize = 10;

impl Model {
    pub fn new(device: &Device) -> Self {
        let input_size = 28 * 28;
        let fc1 = nn::LinearConfig::new(input_size, 128).init(device);
        let fc2 = nn::LinearConfig::new(128, 128).init(device);
        let fc3 = nn::LinearConfig::new(128, NUM_CLASSES).init(device);

        let dropout = nn::DropoutConfig::new(0.25).init();

        Self {
            dropout,
            fc1,
            fc2,
            fc3,
            activation: nn::Gelu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<3>) -> Tensor<2> {
        let [batch_size, height, width] = input.dims();

        let x = input.reshape([batch_size, 1, height, width]).detach();

        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);

        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        self.fc3.forward(x)
    }

    pub fn forward_classification(&self, item: MnistBatch) -> ClassificationOutput {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
    pub fn quantize(self, scheme: QuantScheme) -> Self {
        let calibration = Calibration::MinMax;
        let mut quantizer = Quantizer {
            calibration,
            scheme,
        };

        self.quantize_weights(&mut quantizer)
    }
}

impl TrainStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: MnistBatch) -> TrainOutput<ClassificationOutput> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: MnistBatch) -> ClassificationOutput {
        self.forward_classification(item)
    }
}
