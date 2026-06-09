//! Benchmarks the standalone cubecl `dequantize` kernel in isolation: a
//! quantized tensor materialized back to full precision, with no matmul.
//!
//! This complements `dequantize.rs` (the end-to-end inference benchmark, which
//! is the real use case via the fused dequant-on-read path). Here we time only
//! the dequantization of a representative weight tensor, while still reporting
//! each scheme's model accuracy so we can see the cost/quality trade-off.

use burn::{
    Tensor,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{
        Device, DeviceKind,
        quantization::{QuantLevel, QuantScheme},
    },
};
use burnbench::{
    Benchmark, BenchmarkComputations, BenchmarkDurations, BenchmarkRecord, BenchmarkResult,
    BenchmarkSystemInfo, run_benchmark,
};
use test_mnist::{
    ARTIFACT_DIR,
    inference::{Quality, default_schemes, predictions, prepare_eval, quality},
    model::Model,
};

/// Times `Tensor::dequantize` on the model's quantized hidden-layer weights —
/// the cubecl dequantize kernel, isolated from the matmul. One `execute`
/// dequantizes every quantized weight once, i.e. the per-forward dequant work.
pub struct DequantizeKernelBenchmark {
    device: Device,
    weights: Vec<Tensor<2>>,
    scheme: QuantScheme,
}

impl Benchmark for DequantizeKernelBenchmark {
    type Input = Vec<Tensor<2>>;
    type Output = Vec<Tensor<2>>;

    fn name(&self) -> String {
        let level = match self.scheme.level {
            QuantLevel::Tensor => "tensor".to_string(),
            QuantLevel::Block(b) => format!(
                "block{}",
                b.to_vec()
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            ),
        };
        format!("dequantize-{:?}-{level}", self.scheme.value).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        self.weights.iter().map(|w| w.dims().to_vec()).collect()
    }

    fn num_samples(&self) -> usize {
        TIMING_ITERATIONS
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        input.into_iter().map(|w| w.dequantize()).collect()
    }

    fn prepare(&self) -> Self::Input {
        self.weights.clone()
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

/// Images used to measure accuracy/agreement: the full test set, so the quality
/// estimate is stable and independent of the timing.
const ACCURACY_SAMPLES: usize = 10_000;

/// Number of measured (post-warmup) timing repetitions per scheme.
const TIMING_ITERATIONS: usize = 250;

/// Number of independent timing runs per scheme; the reported timing is the run
/// with the smallest `min`. On a noisy device (e.g. an integrated GPU whose
/// clock fluctuates) the best run approximates the un-throttled performance and
/// is far more reproducible than any single run.
const RUNS: usize = 1;

const BACKEND: BenchBackend = BenchBackend::Wgpu;

// wgpu, cuda, rocm, cpu use cubek dequant kernel
// flex, ndarray, tch, candle use their own impl
#[allow(dead_code)]
#[derive(Clone, Copy)]
enum BenchBackend {
    Wgpu,
    Flex,
    Cpu,
}

impl BenchBackend {
    fn device(self) -> Device {
        match self {
            BenchBackend::Wgpu => Device::wgpu(DeviceKind::DefaultDevice),
            BenchBackend::Flex => Device::flex(),
            BenchBackend::Cpu => Device::cpu(),
        }
    }

    fn name(self) -> &'static str {
        match self {
            BenchBackend::Wgpu => "wgpu",
            BenchBackend::Flex => "flex",
            BenchBackend::Cpu => "cpu",
        }
    }
}

/// A scheme's dequantize-kernel timing together with the quantized model's
/// accuracy/agreement vs full precision.
struct Row {
    /// `None` for the native (unquantized) baseline — nothing to dequantize.
    timing: Option<BenchmarkResult>,
    quality: Quality,
    native: bool,
}

fn bench(device: &Device) -> Vec<Row> {
    let native = Model::new(&device).load_record(
        CompactRecorder::new()
            .load(format!("{ARTIFACT_DIR}/model").into(), &device)
            .expect("Trained model should exist; run train first"),
    );

    // Accuracy baseline (full test set, model predictions). Independent of the
    // dequantize timing below — it just confirms each scheme stays usable.
    let eval = prepare_eval(&native, device, ACCURACY_SAMPLES);

    // Warm up the dequantize kernel before any measured run, using the model's
    // quantized weights under the first scheme.
    let warm_weights = native
        .clone()
        .quantize(default_schemes()[0])
        .quantized_weights();
    for _ in 0..20 {
        for w in &warm_weights {
            let _ = w.clone().dequantize();
        }
    }
    device.sync().unwrap();

    let mut rows = Vec::new();

    // native: accuracy reference only, nothing to dequantize.
    rows.push(Row {
        timing: None,
        quality: quality(&eval.native_pred, &eval),
        native: true,
    });

    for scheme in default_schemes() {
        // Quantize the model once; reuse it for both accuracy and weights.
        let model = native.clone().quantize(scheme);
        let quality = quality(&predictions(&model, eval.images.clone()), &eval);

        // Timing: dequantize the model's quantized hidden-layer weights (the
        // output layer is left in f32 and excluded), best of `RUNS` runs.
        let weights = model.quantized_weights();
        let mut best: Option<BenchmarkResult> = None;
        for _ in 0..RUNS {
            let timing = run_benchmark(DequantizeKernelBenchmark {
                device: device.clone(),
                weights: weights.clone(),
                scheme,
            });
            best = Some(match best {
                Some(prev) if prev.computed.min <= timing.computed.min => prev,
                _ => timing,
            });
        }

        rows.push(Row {
            timing: best,
            quality,
            native: false,
        });
    }

    rows
}

fn main() {
    let device = BACKEND.device();
    let backend_name = BACKEND.name().to_string();
    let feature_name = BACKEND.name();

    let device_name = format!("{:?}", &device);
    let rows = bench(&device);

    println!(
        "\n=== dequantize kernel (model weights, best of {RUNS} runs) vs model accuracy ({ACCURACY_SAMPLES} samples) ==="
    );
    // Format a duration as fixed-decimal milliseconds for stable column widths.
    let ms = |d: std::time::Duration| format!("{:.3}ms", d.as_secs_f64() * 1000.0);

    println!(
        "{:<26} {:>11} {:>11} {:>11} {:>10} {:>11} {:>9}",
        "Scheme", "Median", "Mean", "Min", "Accuracy", "Agreement", "Disagree"
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
        let (name, median, mean, min) = match &row.timing {
            Some(t) => (
                t.name.clone(),
                ms(t.computed.median),
                ms(t.computed.mean),
                ms(t.computed.min),
            ),
            None => (
                "native".to_string(),
                "—".to_string(),
                "—".to_string(),
                "—".to_string(),
            ),
        };
        println!(
            "{:<26} {:>11} {:>11} {:>11} {:>10} {:>11} {:>9}",
            name,
            median,
            mean,
            min,
            format!("{:.2}%", row.quality.accuracy),
            agreement,
            disagree,
        );
    }

    let benches: Vec<BenchmarkResult> = rows.into_iter().filter_map(|row| row.timing).collect();
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
