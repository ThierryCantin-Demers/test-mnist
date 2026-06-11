//! Isolated benchmark for the new `cubek_quant::dequant` kernel.
//!
//! This calls `cubek_quant::dequant::launch_ref` directly on a wgpu cubecl client (burn's
//! `Tensor::dequantize` routes to the *old* `dequantize.rs`, so it can't exercise the new kernel).
//! It times the kernel in isolation (no matmul) and checks correctness via a quantize→dequantize
//! round-trip against the original f32 data.
//!
//! Parameters are fixed to what the new kernel currently asserts:
//!   value `Q4S`, level `Block([8])`, store `PackedU32`, f32 output/scales.

use burnbench::{Benchmark, run_benchmark};
use cubecl::{
    Runtime,
    future::block_on,
    ir::{ElemType, FloatKind},
    prelude::*,
    server::{CopyDescriptor, MemoryLayoutDescriptor, MemoryLayoutStrategy},
    std::tensor::TensorHandle,
    wgpu::WgpuRuntime,
    zspace::shape,
};
use cubek_quant::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};

type R = WgpuRuntime;

// Dictated by the new dequant kernel's assertions.
const VALUE: QuantValue = QuantValue::Q4S;
const BLOCK_SIZE: usize = 8;

// Tensor to dequantize. A single hidden weight matrix of the MLP (4096x4096).
const M: usize = 4096;
const N: usize = 4096;

/// Kernel launches enqueued per timed measurement. The kernel is fast enough that a single
/// launch is dominated by submit/sync overhead, so we run it many times back-to-back and sync
/// once; the reported timings are divided by this to get per-launch cost. Bump it for more
/// stable numbers (at the cost of a longer run).
const INNER_ITERS: usize = 256;

/// The kernel arguments, cloned cheaply (buffer handles) for each timed execution.
#[derive(Clone)]
struct DequantArgs {
    values: TensorBinding<R>,
    scales: TensorBinding<R>,
    output: TensorBinding<R>,
}

struct DequantizeKernelBenchmark {
    client: ComputeClient<R>,
    scheme: QuantScheme,
    values: TensorHandle<R>,
    scales: TensorHandle<R>,
    output: TensorHandle<R>,
    /// Original f32 data and per-block scales, for the correctness check.
    data: Vec<f32>,
    cpu_scales: Vec<f32>,
}

impl Benchmark for DequantizeKernelBenchmark {
    type Input = DequantArgs;
    type Output = ();

    fn name(&self) -> String {
        "dequantize_kernel-q4s-block8".to_string()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![M, N]]
    }

    fn prepare(&self) -> Self::Input {
        DequantArgs {
            values: self.values.clone().binding(),
            scales: self.scales.clone().binding(),
            output: self.output.clone().binding(),
        }
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        // Enqueue INNER_ITERS launches; the surrounding profile syncs once afterwards, so the
        // per-launch overhead is amortized across all of them.
        for _ in 0..INNER_ITERS {
            self.launch_once(&input);
        }
    }

    fn sync(&self) {
        block_on(self.client.sync()).unwrap();
    }
}

mod dequant_kernel {
    pub use cubek_quant::dequant::launch_ref;
}

impl DequantizeKernelBenchmark {
    /// A single dequant launch. Bindings are buffer handles, so cloning them per launch is cheap.
    fn launch_once(&self, args: &DequantArgs) {
        dequant_kernel::launch_ref(
            &self.client,
            args.values.clone(),
            args.output.clone(),
            args.scales.clone(),
            &self.scheme,
            f32::as_type_native_unchecked().storage_type(),
        )
        .unwrap();
    }

    fn new() -> Self {
        let client = R::client(&Default::default());
        let shape = shape![M, N];
        let num_elems = M * N;

        // Synthetic data spread across [-0.5, 0.5).
        let half = num_elems as f32 / 2.0;
        let data: Vec<f32> = (0..num_elems)
            .map(|v| (v as f32 - half) / num_elems as f32)
            .collect();

        // Per-block (last-dim, flattened) calibration scales — symmetric min/max.
        let (q_min, q_max) = VALUE.range();
        let scale_count = num_elems / BLOCK_SIZE;
        let mut cpu_scales = Vec::with_capacity(scale_count);
        for block in 0..scale_count {
            let off = block * BLOCK_SIZE;
            let mut c_max = f32::MIN;
            let mut c_min = f32::MAX;
            for i in 0..BLOCK_SIZE {
                let c = data[off + i];
                c_max = f32::max(c_max, c);
                c_min = f32::min(c_min, c);
            }
            let range = 2.0 * c_min.abs().max(c_max.abs());
            cpu_scales.push(range / (q_max - q_min));
        }

        let scheme = QuantScheme::default()
            .with_level(QuantLevel::block([BLOCK_SIZE as u8]))
            .with_mode(QuantMode::Symmetric)
            .with_value(VALUE)
            .with_store(QuantStore::PackedU32(0))
            .with_param(QuantParam::F32);

        let shape_scale = shape![M, N / BLOCK_SIZE];
        // Packed u32 layout: num_quants values per u32 (== BLOCK_SIZE here).
        let shape_packed = shape![M, N / scheme.num_quants()];

        let input_alloc =
            client.create_tensor_from_slice(f32::as_bytes(&data), shape.clone(), f32::type_size());
        let scale_alloc = client.create_tensor_from_slice(
            f32::as_bytes(&cpu_scales),
            shape_scale.clone(),
            f32::type_size(),
        );

        let input = TensorHandle::<R>::new(
            input_alloc.memory,
            shape.clone(),
            input_alloc.strides,
            f32::as_type_native_unchecked(),
        );
        let scale = TensorHandle::<R>::new(
            scale_alloc.memory,
            shape_scale.clone(),
            scale_alloc.strides,
            f32::as_type_native_unchecked(),
        );

        // Output buffers for quantize: packed values + stored scales.
        let [packed_alloc, out_scale_alloc] = client
            .empty_tensors(vec![
                MemoryLayoutDescriptor {
                    strategy: MemoryLayoutStrategy::Contiguous,
                    shape: shape_packed.clone(),
                    elem_size: u32::type_size(),
                },
                MemoryLayoutDescriptor {
                    strategy: MemoryLayoutStrategy::Contiguous,
                    shape: shape_scale.clone(),
                    elem_size: f32::type_size(),
                },
            ])
            .try_into()
            .unwrap();
        let packed = TensorHandle::<R>::new(
            packed_alloc.memory,
            shape_packed.clone(),
            packed_alloc.strides,
            u32::as_type_native_unchecked(),
        );
        let out_scale = TensorHandle::<R>::new(
            out_scale_alloc.memory,
            shape_scale.clone(),
            out_scale_alloc.strides,
            f32::as_type_native_unchecked(),
        );

        // Quantize once to produce the packed input the dequant kernel consumes.
        cubek_quant::quantize::launch_ref(
            &client,
            input.binding(),
            packed.clone().binding(),
            scale.binding(),
            out_scale.clone().binding(),
            &scheme,
            ElemType::Float(FloatKind::F32),
        )
        .unwrap();

        let output = TensorHandle::<R>::zeros(&client, shape, f32::as_type_native_unchecked());

        Self {
            client,
            scheme,
            values: packed,
            scales: out_scale,
            output,
            data,
            cpu_scales,
        }
    }

    /// Run the kernel once and compare against the original data within quantization tolerance.
    /// Returns `(max_abs_diff, samples_beyond_tolerance)`.
    fn check_correctness(&self) -> (f32, usize) {
        self.launch_once(&self.prepare());
        self.sync();

        let bytes = self.client.read_one_unchecked_tensor(CopyDescriptor::new(
            self.output.handle.clone().binding(),
            self.output.shape().clone(),
            self.output.strides().clone(),
            size_of::<f32>(),
        ));
        let restored = f32::from_bytes(&bytes);

        let rel_tol = 1e-4;
        let mut max_diff = 0f32;
        let mut fails = 0usize;
        for (i, (actual, expected)) in restored.iter().zip(&self.data).enumerate() {
            let scale = self.cpu_scales[i / BLOCK_SIZE];
            let max_error = (scale / 2.0) * (1.0 + rel_tol);
            let diff = (actual - expected).abs();
            max_diff = max_diff.max(diff);
            if diff > max_error {
                fails += 1;
            }
        }
        (max_diff, fails)
    }
}

fn main() {
    let ms = |d: std::time::Duration| format!("{:.4}ms", d.as_secs_f64() * 1000.0);

    let bench = DequantizeKernelBenchmark::new();

    let (max_diff, fails) = bench.check_correctness();
    println!(
        "\n=== dequantize_kernel ({:?}, block {BLOCK_SIZE}, {M}x{N}) ===",
        VALUE
    );
    println!(
        "correctness: {}  (max abs diff {:.6}, {} / {} samples beyond tolerance)",
        if fails == 0 { "PASS" } else { "FAIL" },
        max_diff,
        fails,
        M * N,
    );

    let result = run_benchmark(bench);
    // Each measured sample covers INNER_ITERS launches; report per-launch time.
    let per = |d: std::time::Duration| d / INNER_ITERS as u32;
    println!(
        "timing (per launch, {INNER_ITERS} launches/sample): min {}  median {}  mean {}  (n={})",
        ms(per(result.computed.min)),
        ms(per(result.computed.median)),
        ms(per(result.computed.mean)),
        result.raw.durations.len(),
    );
}
