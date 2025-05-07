// from https://github.com/eugenehp/gpu-fft/blob/master/src/fft.rs
use cubecl::prelude::*;
use once_cell::sync::OnceCell;

use std::f32::consts::PI;
static FFT_PRELOADED: OnceCell<()> = OnceCell::new();
// The general advice for WebGPU is to choose a workgroup size of 64
// Common sizes are 32, 64, 128, 256, or 512 threads per workgroup.
// Apple Metal supports a maximum workgroup size of 1024 threads.
pub(crate) const WORKGROUP_SIZE: u32 = 1024;
/// Performs a Fast Fourier Transform (FFT) on the input data.
///
/// This kernel computes the FFT of a given input array of complex numbers represented as
/// separate real and imaginary parts. The FFT is computed using the Cooley-Tukey algorithm,
/// which is efficient for large datasets.
///
/// # Parameters
///
/// - `input`: An array of complex numbers represented as lines of type `Line<F>`, where `F`
///   is a floating-point type. The input array should contain `n` complex numbers.
/// - `real_output`: A mutable reference to an array of lines where the real parts of the
///   FFT output will be stored.
/// - `imag_output`: A mutable reference to an array of lines where the imaginary parts of
///   the FFT output will be stored.
/// - `n`: The number of complex samples in the input array. This value is provided at compile-time.
///
/// # Safety
///
/// This function is marked as `unsafe` because it performs raw pointer operations and assumes
/// that the input and output arrays are correctly sized and aligned. The caller must ensure
/// that the input data is valid and that the output arrays have sufficient space to store
/// the results.
///
/// # Example
///
/// ```rust
/// let input = vec![1.0, 0.0, 0.0, 0.0]; // Example input
/// let (real, imag) = fft::<YourRuntimeType>(device, input);
/// ```
///
/// # Returns
///
/// This function does not return a value directly. Instead, it populates the `output` array
/// with the real and imaginary parts of the FFT result interleaved.
#[cube(launch)]
fn fft_kernel<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>, #[comptime] n: u32) {
    let idx = ABSOLUTE_POS;
    if idx < n {
        let mut real = Line::<F>::new(F::new(0.0));
        let mut imag = Line::<F>::new(F::new(0.0));
        // Precompute the angle increment
        let angle_increment = -2.0 * PI / n as f32;
        // #[unroll(true)]
        for k in 0..n {
            let angle = F::cast_from(angle_increment) * F::cast_from(k) * F::cast_from(idx);
            let (cos_angle, sin_angle) = (F::cos(angle), F::sin(angle));

            // Combine the multiplication and addition
            real += input[k] * Line::new(cos_angle);
            imag += input[k] * Line::new(sin_angle);
        }

        // Store the real and imaginary parts in an interleaved manner
        output[idx * 2] = Line::new(F::cast_from(real)); // Real part
        output[idx * 2 + 1] = Line::new(F::cast_from(imag)); // Imaginary part
    }
}

/// Computes the Fast Fourier Transform (FFT) for a batch of input vectors.
///
/// This function launches a single kernel call that performs FFTs over multiple input vectors.
/// It assumes that each input vector is the same length.
///
/// # Parameters
/// - `device`: Reference to the WGPU device.
/// - `inputs`: A slice of input vectors, where each vector is an independent FFT input.
///
/// # Returns
/// A vector of tuples, each containing the real and imaginary parts of the FFT output.
pub fn fft_batched<R: Runtime>(
    device: &R::Device,
    inputs: &[Vec<f32>],
) -> Vec<(Vec<f32>, Vec<f32>)> {
    assert!(!inputs.is_empty(), "Input batch cannot be empty");
    let client = R::client(device);
    let n = inputs[0].len();
    let batch_size = inputs.len();

    for input in inputs.iter() {
        assert_eq!(input.len(), n, "All inputs must have the same length");
    }

    // Flatten input: [batch][n] -> [batch * n]
    let flat_input: Vec<f32> = inputs.iter().flat_map(|v| v.iter()).copied().collect();
    let input_handle = client.create(f32::as_bytes(&flat_input));
    let output_handle = client.empty(batch_size * n * 2 * core::mem::size_of::<f32>());

    // Pre-create and reuse the kernel launch configuration
    let num_workgroups = ((n as u32 * batch_size as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let cube_count = CubeCount::Static(num_workgroups, 1, 1);
    let cube_dim = CubeDim::new(WORKGROUP_SIZE, 1, 1);

    unsafe {
        fft_kernel::launch::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&input_handle, n * batch_size, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, n * 2 * batch_size, 1),
            n as u32,
        );
    }

    let output_bytes = client.read_one(output_handle.binding());
    let output = f32::from_bytes(&output_bytes);

    let mut results = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let base = b * n * 2;
        let real: Vec<f32> = (0..n).map(|i| output[base + 2 * i]).collect();
        let imag: Vec<f32> = (0..n).map(|i| output[base + 2 * i + 1]).collect();
        results.push((real, imag));
    }

    results
}
pub fn preload_fft_kernel<R: Runtime>(device: &R::Device, n: usize) {
    let client = R::client(device);

    let dummy_input = vec![0.0f32; n];
    let input_handle = client.create(f32::as_bytes(&dummy_input));
    let output_handle = client.empty(n * 2 * std::mem::size_of::<f32>());

    let num_workgroups = (n as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    unsafe {
        fft_kernel::launch::<f32, R>(
            &client,
            CubeCount::Static(num_workgroups, 1, 1),
            CubeDim::new(WORKGROUP_SIZE, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, n * 2, 1),
            n as u32,
        );
    }
}
/// Computes the Fast Fourier Transform (FFT) of a vector of f32 input data.
///
/// This function initializes the FFT computation on the provided input vector, launching
/// the FFT kernel to perform the transformation. The input data is expected to be in the
/// form of real numbers, which are treated as complex numbers with zero imaginary parts.
///
/// # Parameters
///
/// - `device`: A reference to the device on which the FFT computation will be performed.
/// - `input`: A vector of `f32` values representing the real parts of the input data.
///
/// # Returns
///
/// A tuple containing two vectors:
/// - A vector of `f32` values representing the real parts of the FFT output.
/// - A vector of `f32` values representing the imaginary parts of the FFT output.
///
/// # Example
///
/// ```rust
/// let input = vec![1.0, 0.0, 0.0, 0.0]; // Example input
/// let (real, imag) = fft::<YourRuntimeType>(device, input);
/// ```
///
/// # Safety
///
/// This function uses unsafe operations to interact with the underlying runtime and device.
/// The caller must ensure that the input data is valid and that the device is properly set up
/// for computation.
pub fn fft<R: Runtime>(device: &R::Device, input: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    let client = R::client(device);
    let n = input.len();

    FFT_PRELOADED.get_or_init(|| {
        preload_fft_kernel::<R>(device, n);
    });

    let input_handle = client.create(f32::as_bytes(&input));
    let output_handle = client.empty(n * 2 * core::mem::size_of::<f32>()); // Adjust for interleaved output

    let num_workgroups = (n as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    unsafe {
        fft_kernel::launch::<f32, R>(
            &client,
            CubeCount::Static(num_workgroups, 1, 1),
            CubeDim::new(WORKGROUP_SIZE, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, n * 2, 1), // Adjust for interleaved output
            n as u32,
        )
    };

    let output_bytes = client.read_one(output_handle.binding());
    let output = f32::from_bytes(&output_bytes);

    // Split the interleaved output into real and imaginary parts
    let real: Vec<f32> = output.iter().step_by(2).cloned().collect();
    let imag: Vec<f32> = output.iter().skip(1).step_by(2).cloned().collect();

    // println!(
    //     "real {:?}..{:?}",
    //     &real[0..10],
    //     &real[real.len() - 10..real.len() - 1]
    // );
    // println!(
    //     "imag {:?}..{:?}",
    //     &imag[0..10],
    //     &imag[imag.len() - 10..imag.len() - 1]
    // );

    (real, imag)
}

use std::sync::Arc;

pub struct FftBatcher<R: Runtime> {
    buffer: Vec<Vec<f32>>,
    device: Arc<R::Device>,
    n: usize,
    capacity: usize,
}

impl<R: Runtime> FftBatcher<R> {
    pub fn new(device: Arc<R::Device>, n: usize, capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            device,
            n,
            capacity,
        }
    }

    /// Push a new frame into the batcher. Returns Some(results) when the buffer reaches capacity.
    pub fn push(&mut self, frame: Vec<f32>) -> Option<Vec<(Vec<f32>, Vec<f32>)>> {
        assert_eq!(frame.len(), self.n, "Frame size mismatch");
        self.buffer.push(frame);
        if self.buffer.len() == self.capacity {
            let results = fft_batched::<R>(&self.device, &self.buffer);
            self.buffer.clear();
            Some(results)
        } else {
            None
        }
    }

    /// Force flush the current buffer if it's not empty.
    pub fn flush(&mut self) -> Option<Vec<(Vec<f32>, Vec<f32>)>> {
        if self.buffer.is_empty() {
            None
        } else {
            let results = fft_batched::<R>(&self.device, &self.buffer);
            self.buffer.clear();
            Some(results)
        }
    }
}
