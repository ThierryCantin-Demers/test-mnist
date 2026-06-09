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
    activation: nn::Gelu,
    layers: Vec<nn::Linear>,
}

const NUM_CLASSES: usize = 10;
const INPUT_SIZE: usize = 28 * 28;
const NUM_LAYERS: usize = 4;
const LAYER_SIZE: usize = 4096;

impl Model {
    pub fn new(device: &Device) -> Self {
        const {
            assert!(NUM_LAYERS >= 2);
        }
        let mut layers = Vec::new();
        let mut prev_size = INPUT_SIZE;

        for _ in 0..NUM_LAYERS {
            let fc = nn::LinearConfig::new(prev_size, LAYER_SIZE).init(device);
            layers.push(fc);
            prev_size = LAYER_SIZE;
        }

        let fc = nn::LinearConfig::new(prev_size, NUM_CLASSES).init(device);
        layers.push(fc);

        let dropout = nn::DropoutConfig::new(0.25).init();

        let activation = nn::Gelu::new();

        Self {
            dropout,
            layers,
            activation,
        }
    }

    pub fn forward(&self, input: Tensor<3>) -> Tensor<2> {
        let [batch_size, height, width] = input.dims();

        let x = input.reshape([batch_size, 1, height, width]).detach();

        let [batch_size, channels, height, width] = x.dims();
        let mut x = x.reshape([batch_size, channels * height * width]);

        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i != last {
                x = self.activation.forward(x);
                x = self.dropout.forward(x);
            }
        }

        x
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

    /// Weight tensors of the quantized hidden layers only, in order.
    ///
    /// The output layer (10-wide) is left in full precision by [`quantize`] and
    /// is deliberately excluded — `[..last]` drops it.
    ///
    /// [`quantize`]: Model::quantize
    pub fn quantized_weights(&self) -> Vec<Tensor<2>> {
        let last = self.layers.len() - 1;
        self.layers[..last]
            .iter()
            .map(|layer| layer.weight.val())
            .collect()
    }

    pub fn quantize(mut self, scheme: QuantScheme) -> Self {
        let mut quantizer = Quantizer {
            calibration: Calibration::MinMax,
            scheme,
        };

        self.activation = self.activation.quantize_weights(&mut quantizer);
        self.dropout = self.dropout.quantize_weights(&mut quantizer);

        let output = self.layers.pop().expect("model has at least one layer");
        let mut layers: Vec<nn::Linear> = self
            .layers
            .into_iter()
            .map(|layer| layer.quantize_weights(&mut quantizer))
            .collect();
        layers.push(output);
        self.layers = layers;

        self
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
