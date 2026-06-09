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
    ARTIFACT_DIR,
    data::MnistBatcher,
    inference::{Quality, default_schemes, predictions, prepare_eval, quality},
    inference_device,
    model::Model,
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
                    QuantLevel::Block(b) => {
                        format!(
                            "block{}",
                            b.to_vec()
                                .into_iter()
                                .map(|s| s.to_string())
                                .collect::<Vec<_>>()
                                .join("x")
                        )
                    }
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
        predictions(&self.model, input)
    }

    fn prepare(&self) -> Self::Input {
        let dataset = MnistDataset::test();
        let items: Vec<_> = (0..self.samples).filter_map(|i| dataset.get(i)).collect();
        let batch = MnistBatcher::default().batch(items, &self.device);
        let images = batch.images;
        images
    }

    fn num_samples(&self) -> usize {
        TIMING_ITERATIONS
    }

    fn sync(&self) {
        self.device.sync().expect("Should sync without error");
    }
}

/// Images per forward pass in the timing loop. Kept small so the matmul is
/// weight-bandwidth-bound — that's the regime where the dequant-on-read cost of
/// each scheme actually shows up (a large batch is compute-bound and hides it).
const TIMING_BATCH: usize = 256;

/// Images used to measure accuracy/agreement: the full test set, so the quality
/// estimate is stable and independent of the (small) timing batch.
const ACCURACY_SAMPLES: usize = 10_000;

/// Number of measured (post-warmup) timing repetitions per scheme.
const TIMING_ITERATIONS: usize = 100;

/// One model's timing together with its accuracy/agreement vs full precision.
struct Row {
    timing: BenchmarkResult,
    quality: Quality,
    native: bool,
}

fn bench(device: &Device) -> Vec<Row> {
    let native = Model::new(&device).load_record(
        CompactRecorder::new()
            .load(format!("{ARTIFACT_DIR}/model").into(), &device)
            .expect("Trained model should exist; run train first"),
    );

    // Shared test batch + full-precision baseline predictions, computed once.
    // These accuracy forwards are separate from `run_benchmark`'s timing loop
    // (whose outputs are discarded), so they don't affect the measured timings.
    let eval = prepare_eval(&native, device, ACCURACY_SAMPLES);
    let mut rows = Vec::new();

    // native: reuse the baseline predictions, no extra forward pass
    let quality_native = quality(&eval.native_pred, &eval);
    let timing = run_benchmark(DequantizeBenchmark {
        quant_scheme: None,
        model: native.clone(),
        device: device.clone(),
        samples: TIMING_BATCH,
    });
    rows.push(Row {
        timing,
        quality: quality_native,
        native: true,
    });

    // quantized variants
    for scheme in default_schemes() {
        let model = native.clone().quantize(scheme);
        let quality = quality(&predictions(&model, eval.images.clone()), &eval);
        let timing = run_benchmark(DequantizeBenchmark {
            quant_scheme: Some(scheme),
            model,
            device: device.clone(),
            samples: TIMING_BATCH,
        });
        rows.push(Row {
            timing,
            quality,
            native: false,
        });
    }

    rows
}

fn main() {
    let device = inference_device();
    let backend_name = "wgpu".to_string();
    let feature_name = "wgpu";

    let device_name = format!("{:?}", &device);
    let rows = bench(&device);

    println!(
        "\n=== quantization: timing (batch {TIMING_BATCH}) vs accuracy ({ACCURACY_SAMPLES} samples) ==="
    );
    println!(
        "{:<26} {:>13} {:>13} {:>10} {:>11} {:>9}",
        "Scheme", "Median", "Mean", "Accuracy", "Agreement", "Disagree"
    );
    for row in &rows {
        let (agreement, disagree) = if row.native {
            ("—".to_string(), "—".to_string())
        } else {
            (
                format!("{:.2}%", row.quality.agreement),
                row.quality.disagreements.to_string(),
            )
        };
        println!(
            "{:<26} {:>11.8}ms {:>11.8}ms {:>10} {:>11} {:>9}",
            row.timing.name,
            format!("{:?}", row.timing.computed.median),
            format!("{:?}", row.timing.computed.mean),
            format!("{:.2}%", row.quality.accuracy),
            agreement,
            disagree,
        );
    }

    let benches: Vec<BenchmarkResult> = rows.into_iter().map(|row| row.timing).collect();
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
