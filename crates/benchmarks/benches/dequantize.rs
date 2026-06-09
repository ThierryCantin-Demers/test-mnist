use burn::{
    Tensor,
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, vision::MnistDataset},
    },
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{
        Device, Int,
        quantization::{QuantLevel, QuantScheme},
    },
};
use burnbench::{
    Benchmark, BenchmarkComputations, BenchmarkDurations, BenchmarkRecord, BenchmarkResult,
    BenchmarkSystemInfo, run_benchmark,
};
use test_mnist::{
    ARTIFACT_DIR, data::MnistBatcher, inference::default_schemes, inference_device, model::Model,
};

pub struct DequantizeBenchmark {
    device: Device,
    model: Model,
    quant_scheme: Option<QuantScheme>,
    samples: usize,
}

impl Benchmark for DequantizeBenchmark {
    type Input = Tensor<3>;
    type Output = Tensor<1, Int>;

    fn name(&self) -> String {
        match &self.quant_scheme {
            Some(scheme) => {
                let level = match scheme.level {
                    QuantLevel::Tensor => "tensor".to_string(),
                    QuantLevel::Block(b) => format!("block{}", b.num_elements()),
                };
                format!("dequantize-{:?}-{level}", scheme.value).to_lowercase()
            }
            None => "dequantize-native".to_string(),
        }
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.samples, 28, 28]]
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        self.model.forward(input).argmax(1).flatten::<1>(0, 1)
    }

    fn prepare(&self) -> Self::Input {
        let dataset = MnistDataset::test();
        let items: Vec<_> = (0..self.samples).filter_map(|i| dataset.get(i)).collect();
        let batch = MnistBatcher::default().batch(items, &self.device);
        let images = batch.images;
        images
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

struct Config {
    quant_scheme: Option<QuantScheme>,
}

const NUM_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    let native = Model::new(&device).load_record(
        CompactRecorder::new()
            .load(format!("{ARTIFACT_DIR}/model").into(), &device)
            .expect("Trained model should exist; run train first"),
    );

    // native
    let benchmark = DequantizeBenchmark {
        quant_scheme: None,
        model: native.clone(),
        device: device.clone(),
        samples: NUM_SAMPLES,
    };
    let result = run_benchmark(benchmark);
    results.push(result);

    // quantized
    let quant_schemes = default_schemes().into_iter().map(|scheme| Config {
        quant_scheme: Some(scheme),
    });

    for config in quant_schemes {
        let model = if let Some(scheme) = &config.quant_scheme {
            native.clone().quantize(*scheme)
        } else {
            native.clone()
        };
        let benchmark = DequantizeBenchmark {
            quant_scheme: config.quant_scheme.clone(),
            model,
            device: device.clone(),
            samples: NUM_SAMPLES,
        };
        let result = run_benchmark(benchmark);
        results.push(result);
    }

    results
}

fn main() {
    let device = inference_device();
    let backend_name = "wgpu".to_string();
    let feature_name = "wgpu";

    let device_name = format!("{:?}", &device);
    let benches = bench(&device);

    println!("\n=== dequantize benchmarks ({NUM_SAMPLES} samples) ===");
    println!(
        "{:<32} {:>12} {:>12} {:>12}",
        "Benchmark", "Median", "Mean", "Min"
    );
    for b in &benches {
        println!(
            "{:<32} {:>12} {:>12} {:>12}",
            b.name,
            format!("{:?}", b.computed.median),
            format!("{:?}", b.computed.mean),
            format!("{:?}", b.computed.min),
        );
    }

    __save_result(benches, backend_name, device_name, None, None, feature_name);
}

pub fn __save_result(
    benches: Vec<BenchmarkResult>,
    backend_name: String,
    device: String,
    url: Option<&str>,
    token: Option<&str>,
    feature: &str,
) {
    let burn_version =
        std::env::var("BURN_BENCH_BURN_VERSION").unwrap_or_else(|_| "main".to_string());

    let records: Vec<BenchmarkRecord> = benches
        .into_iter()
        .map(|bench| BenchmarkRecord {
            backend: backend_name.clone(),
            device: device.clone(),
            feature: feature.to_string(),
            burn_version: burn_version.clone(),
            system_info: BenchmarkSystemInfo::new(),
            results: BenchmarkResult {
                raw: BenchmarkDurations {
                    timing_method: Default::default(),
                    durations: bench.raw.durations,
                },
                computed: BenchmarkComputations {
                    mean: bench.computed.mean,
                    median: bench.computed.median,
                    variance: bench.computed.variance,
                    min: bench.computed.min,
                    max: bench.computed.max,
                },
                git_hash: bench.git_hash,
                name: bench.name,
                options: bench.options,
                shapes: bench.shapes,
                timestamp: bench.timestamp,
            },
        })
        .collect();

    burnbench::save_records(records, url, token).unwrap()
}
