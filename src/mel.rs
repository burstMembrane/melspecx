use crate::colors;
use once_cell::sync::Lazy;
use std::collections::HashMap;
static GPU_DEVICE: Lazy<WgpuDevice> = Lazy::new(|| Default::default());
static FILTERBANKS: Lazy<HashMap<(usize, u32, usize), Array2<f32>>> = Lazy::new(|| {
    let mut m = HashMap::new();
    let n_fft = 1024;
    let n_freqs = n_fft / 2 + 1;
    for &sample_rate in &[44100.0f32, 22050.0, 16000.0, 8000.0] {
        let f_min = 0.0;
        let f_max = sample_rate / 2.0;
        for &n_mels in &[96usize, 128usize] {
            let fbanks = mel_filter_bank(n_freqs, f_min, f_max, n_mels, sample_rate);
            let fbanks_flat: Vec<f32> = fbanks.into_iter().flatten().collect();
            let arr = Array2::from_shape_vec((n_mels, n_freqs), fbanks_flat).unwrap();
            m.insert((n_fft, sample_rate as u32, n_mels), arr);
        }
    }
    m
});
use cubecl::wgpu::WgpuDevice;
use gpu_fft::fft::fft as gpu_fft_fft;
use image::ImageBuffer;
use image::Rgb;
use ndarray::Array2;
use num::complex::Complex;
use std::time::Instant;

// Only include this when the python-bindings feature is enabled
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
use pyo3::types::PyBytes;

use crate::audio::read_wav;
use rayon::prelude::*;
use std::default::Default;
use std::io::Cursor;
type Runtime = cubecl::wgpu::WgpuRuntime;
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn mel_spec_from_path(
    path: String,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    colormap: String,
    width_px: u32,
    height_px: u32,
    top_db: f32,
) -> PyResult<Py<PyAny>> {
    // log start time to python
    let start_time = Instant::now();
    let cmap = colors::Colormap::from_name(&colormap).unwrap();
    let (audio, sr) = read_wav(path, Some(true)).unwrap();
    println!("Audio read time: {:?}", start_time.elapsed());
    let start_time = Instant::now();
    let mel_spec = mel_spectrogram_db(
        &MelConfig::new(
            sr as f32,
            n_fft,
            win_length,
            hop_length,
            f_min,
            f_max,
            n_mels,
            top_db,
            SpectrogramConfig::new(true),
        ),
        audio,
    );
    println!(
        "Mel spectrogram generation time: {:?}",
        start_time.elapsed()
    );

    let start_time = Instant::now();
    let image = plot_mel_spec(mel_spec, cmap, width_px, height_px);
    println!("Plotting time: {:?}", start_time.elapsed());

    let start_time = Instant::now();
    let result = Python::with_gil(|py| {
        // Convert image to PNG bytes
        let mut buffer = Cursor::new(Vec::new());
        match image.write_to(&mut buffer, image::ImageFormat::Png) {
            Ok(_) => {
                // Convert to Python bytes object
                let bytes = PyBytes::new(py, &buffer.into_inner());
                Ok(bytes.into())
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to encode image: {}",
                e
            ))),
        }
    });
    println!("Image encoding time: {:?}", start_time.elapsed());
    result
}

fn gpu_spectrogram(
    waveform: Vec<f32>,
    n_fft: usize,
    _win_length: usize,
    _hop_length: usize,
    onesided: bool,
) -> Vec<Vec<f32>> {
    let start_time = Instant::now();

    let device = (*GPU_DEVICE).clone();

    println!("CubeCL WGPU device initialized");
    println!("Using device type: WGPU");
    println!("GPU device initialization time: {:?}", start_time.elapsed());

    let start_time = Instant::now();
    let chunk_size = n_fft * 2;

    // Precompute all padded frames
    let frames: Vec<Vec<f32>> = waveform
        .chunks(chunk_size)
        .map(|chunk| {
            let mut frame = vec![0.0f32; chunk_size];
            frame[..chunk.len()].copy_from_slice(chunk);
            frame
        })
        .collect();

    let total_ffts = frames.len();
    let fft_start = Instant::now();
    let spec: Vec<Vec<f32>> = frames
        .into_par_iter()
        .map(|frame| {
            let (real, imag) = gpu_fft_fft::<Runtime>(&device, frame);
            let half = if onesided { n_fft / 2 + 1 } else { n_fft };
            (0..half)
                .map(|i| (real[i].powi(2) + imag[i].powi(2)).sqrt())
                .collect()
        })
        .collect();
    let fft_time = fft_start.elapsed();

    let chunks_processing_time = start_time.elapsed();
    println!(
        "GPU spectrogram total processing time: {:?}",
        chunks_processing_time
    );
    println!(
        "Average FFT time: {:?}",
        fft_time.div_f32(total_ffts as f32)
    );
    println!("Total FFT operations: {}", total_ffts);
    println!("GPU spectrogram chunks count: {}", waveform.len() / n_fft);
    println!("GPU spectrogram time: {:?}", start_time.elapsed());
    spec
}

#[derive(Clone)]
#[cfg_attr(feature = "python-bindings", derive(IntoPyObject, IntoPyObjectRef))]
pub struct SpectrogramConfig {
    onesided: bool,
}
impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self { onesided: true }
    }
}
impl SpectrogramConfig {
    pub fn new(onesided: bool) -> Self {
        Self { onesided }
    }
}

// derive a struct from the MelConfig struct
#[derive(Clone)]
#[cfg_attr(feature = "python-bindings", derive(IntoPyObjectRef))]
pub struct MelConfig {
    sample_rate: f32,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    top_db: f32,
    spectrogram_config: SpectrogramConfig,
}

impl MelConfig {
    pub fn new(
        sample_rate: f32,
        n_fft: usize,
        win_length: usize,
        hop_length: usize,
        f_min: f32,
        f_max: f32,
        n_mels: usize,
        top_db: f32,
        spectrogram_config: SpectrogramConfig,
    ) -> Self {
        Self {
            sample_rate,
            n_fft,
            win_length,
            hop_length,
            f_min,
            f_max,
            n_mels,
            top_db,
            spectrogram_config,
        }
    }
}
pub fn mel_spectrogram_db(config: &MelConfig, waveform: Vec<f32>) -> Vec<Vec<f32>> {
    let start_time = Instant::now();
    let mel_spec: Vec<Vec<f32>> = mel_spectrogram(config, waveform);
    println!(
        "Mel spectrogram computation time: {:?}",
        start_time.elapsed()
    );
    mel_spec
}

fn mel_spectrogram(config: &MelConfig, waveform: Vec<f32>) -> Vec<Vec<f32>> {
    let start_time = Instant::now();
    let spectrogram = spectrogram(
        waveform,
        config.n_fft,
        config.win_length,
        config.hop_length,
        config.spectrogram_config.onesided,
    );
    println!("Spectrogram generation time: {:?}", start_time.elapsed());

    let start_time = Instant::now();
    let frames = spectrogram.len();
    let n_freqs = spectrogram[0].len();
    let spec_flat: Vec<f32> = spectrogram.into_iter().flatten().collect();
    let spec_arr = Array2::from_shape_vec((frames, n_freqs), spec_flat).unwrap();

    // Use cached filterbank if available, otherwise compute dynamically
    let fbanks_arr: Array2<f32> = if let Some(arr) =
        FILTERBANKS.get(&(config.n_fft, config.sample_rate as u32, config.n_mels))
    {
        arr.clone()
    } else {
        let n_freqs = config.n_fft / 2 + 1;
        let fbanks = mel_filter_bank(
            n_freqs,
            config.f_min,
            config.f_max,
            config.n_mels,
            config.sample_rate,
        );
        let fbanks_flat: Vec<f32> = fbanks.into_iter().flatten().collect();
        Array2::from_shape_vec((config.n_mels, n_freqs), fbanks_flat).unwrap()
    };
    let mel_arr = spec_arr.dot(&fbanks_arr.t());

    // Fuse amplitude→dB conversion
    let epsilon = 1e-10f32;
    let result: Vec<Vec<f32>> = mel_arr
        .mapv(|x| 20.0 * x.max(epsilon).log10())
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    println!(
        "Mel filtering + dB conversion time: {:?}",
        start_time.elapsed()
    );
    result
}

fn spectrogram(
    waveform: Vec<f32>,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    onesided: bool,
) -> Vec<Vec<f32>> {
    let start_time = Instant::now();
    // GPU-accelerated FFT
    let result = gpu_spectrogram(waveform, n_fft, win_length, hop_length, onesided);
    println!(
        "Total spectrogram function time: {:?}",
        start_time.elapsed()
    );
    result
}

// we might reimplement this in the future
#[allow(dead_code)]
fn fft(input: Vec<f32>, n_fft: usize) -> Vec<Complex<f32>> {
    let num_samples = input.len();
    assert!(n_fft.is_power_of_two(), "n_fft must be a power of 2");
    assert!(
        num_samples <= n_fft,
        "n must be less than or equal to n_fft"
    );

    if num_samples <= 1 {
        return input.into_iter().map(|x| Complex::new(x, 0.0)).collect();
    }

    let padded_input = if num_samples < n_fft {
        let padding = vec![0.0; n_fft - num_samples];
        input.clone().into_iter().chain(padding).collect()
    } else {
        input.clone()
    };

    // Split into even and odd parts
    let even: Vec<f32> = padded_input.iter().step_by(2).cloned().collect();
    let odd: Vec<f32> = padded_input.iter().skip(1).step_by(2).cloned().collect();

    // Recursive FFT on even and odd parts
    let even_fft = fft(even, n_fft / 2);
    let odd_fft = fft(odd, n_fft / 2);

    // Combine results
    let mut output = vec![Complex::new(0.0, 0.0); n_fft];
    for k in 0..(n_fft / 2) {
        let t = odd_fft[k]
            * Complex::from_polar(1.0, -2.0 * std::f32::consts::PI * k as f32 / n_fft as f32);
        output[k] = even_fft[k] + t;
        output[k + n_fft / 2] = even_fft[k] - t; // Exploit symmetry
    }
    output
}
#[allow(dead_code)]
fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (length - 1) as f32).cos()))
        .collect()
}

fn mel_filter_bank(
    n_freqs: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    sample_rate: f32,
) -> Vec<Vec<f32>> {
    let f_nyquist = sample_rate / 2.0;

    let all_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| f_nyquist * i as f32 / (n_freqs - 1) as f32)
        .collect(); // (n_freqs,)

    let m_min = hz_to_mel(f_min);
    let m_max = hz_to_mel(f_max);

    let m_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| m_min + (m_max - m_min) * i as f32 / (n_mels + 1) as f32)
        .collect(); // (n_mels + 2,)

    let f_points: Vec<f32> = m_points.iter().map(|&mel| mel_to_hz(mel)).collect();

    let f_diff: Vec<f32> = f_points
        .iter()
        .skip(1)
        .zip(f_points.iter().take(f_points.len() - 1))
        .map(|(f2, f1)| f2 - f1)
        .collect(); // (n_mels + 1,)

    let slopes: Vec<Vec<f32>> = all_freqs
        .iter()
        .map(|&f| f_points.iter().map(|&fp| fp - f).collect())
        .collect(); // (n_freqs, n_mels + 2)

    let down_slopes: Vec<Vec<f32>> = slopes
        .iter()
        .map(|slope_slice| {
            slope_slice
                .iter()
                .take(n_mels)
                .zip(f_diff.iter().take(n_mels))
                .map(|(slope, &diff)| -1.0 * slope / diff)
                .collect()
        })
        .collect(); // (n_freqs, n_mels)

    let up_slopes: Vec<Vec<f32>> = slopes
        .iter()
        .map(|slope_slice| {
            slope_slice
                .iter()
                .skip(2)
                .take(n_mels)
                .zip(f_diff.iter().skip(1).take(n_mels))
                .map(|(slope, &diff)| slope / diff)
                .collect()
        })
        .collect();

    let mut fbanks: Vec<Vec<f32>> = up_slopes
        .iter()
        .zip(down_slopes.iter())
        .map(|(up, down)| {
            let row = down
                .iter()
                .zip(up.iter())
                .map(|(&d, &u)| d.min(u).max(0.0)) // Use both up and down slopes
                .collect::<Vec<f32>>();
            row
        })
        .collect();

    // Apply Slaney normalization
    for i in 0..n_mels {
        let enorm = 2.0 / (f_points[i + 2] - f_points[i]);
        for fbank in fbanks.iter_mut() {
            fbank[i] *= enorm;
        }
    }

    fbanks
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}
pub fn plot_mel_spec(
    mel_spec: Vec<Vec<f32>>,
    cmap: colors::Colormap,
    width_px: u32,
    height_px: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let time_steps = mel_spec.len();
    let mel_bands = mel_spec[0].len();

    // Find min and max values for normalization
    let flat_vals: Vec<f32> = mel_spec
        .iter()
        .flat_map(|row| row.iter().cloned())
        .collect();
    let smin = flat_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let smax = flat_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let color_map = colors::precompute_colormap(&cmap);

    let mut image = ImageBuffer::new(width_px, height_px);

    // Use bilinear interpolation to avoid blocky artifacts
    for px in 0..width_px {
        // Convert pixel to time position (as a float for interpolation)
        let time_pos = (px as f32) * (time_steps as f32 - 1.0) / (width_px as f32 - 1.0);
        let time_idx = time_pos.floor() as usize;
        let time_frac = time_pos - time_idx as f32;

        // Handle edge case
        let time_idx_next = if time_idx >= time_steps - 1 {
            time_steps - 1
        } else {
            time_idx + 1
        };

        for py in 0..height_px {
            // Convert pixel to mel band position (as a float for interpolation)
            // Note the reversed y-axis (height_px - 1 - py)
            let mel_pos =
                ((height_px - 1 - py) as f32) * (mel_bands as f32 - 1.0) / (height_px as f32 - 1.0);
            let mel_idx = mel_pos.floor() as usize;
            let mel_frac = mel_pos - mel_idx as f32;

            // Handle edge case
            let mel_idx_next = if mel_idx >= mel_bands - 1 {
                mel_bands - 1
            } else {
                mel_idx + 1
            };

            // Bilinear interpolation between four nearest points
            let val_tl = mel_spec[time_idx][mel_idx];
            let val_tr = mel_spec[time_idx_next][mel_idx];
            let val_bl = mel_spec[time_idx][mel_idx_next];
            let val_br = mel_spec[time_idx_next][mel_idx_next];

            let val_top = val_tl * (1.0 - time_frac) + val_tr * time_frac;
            let val_bottom = val_bl * (1.0 - time_frac) + val_br * time_frac;
            let val = val_top * (1.0 - mel_frac) + val_bottom * mel_frac;

            // Convert to color
            if val.is_finite() && smin != smax {
                let norm = ((val - smin) / (smax - smin)).clamp(0.0, 1.0);
                let idx = (norm * 255.0).round() as usize;
                image.put_pixel(px, py, Rgb(color_map[idx.min(255)]));
            } else {
                image.put_pixel(px, py, Rgb([0, 0, 0])); // fallback for NaNs or constant images
            }
        }
    }

    image
}

fn amplitude_to_db(amplitudes: Vec<Vec<f32>>, top_db: f32) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    let dbs: Vec<Vec<f32>> = amplitudes
        .into_par_iter()
        .map(|row| {
            row.into_iter()
                .map(|amp| 20.0 * amp.max(1e-10).log10())
                .collect()
        })
        .collect();

    let max_db = dbs
        .iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let clipped: Vec<Vec<f32>> = dbs
        .into_par_iter()
        .map(|row| row.into_iter().map(|db| db.max(max_db - top_db)).collect())
        .collect();

    clipped
}

fn _assert_complex_eq(left: Complex<f32>, right: Complex<f32>) {
    // tolerance for floating-point comparison
    const EPSILON: f32 = 1e-5;
    assert!(
        (left.re - right.re).abs() < EPSILON,
        "left: {:?}, right: {:?}",
        left,
        right
    );
    assert!(
        (left.im - right.im).abs() < EPSILON,
        "left: {:?}, right: {:?}",
        left,
        right
    );
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn mel_spectrogram_db_py(config: MelConfig, waveform: Vec<f32>) -> Vec<Vec<f32>> {
    let start_time = Instant::now();
    let result = mel_spectrogram_db(&config, waveform);
    println!(
        "mel_spectrogram_db_py execution time: {:?}",
        start_time.elapsed()
    );
    result
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn plot_mel_spec_py(
    mel_spec: Vec<Vec<f32>>,
    cmap: colors::Colormap,
    width_px: u32,
    height_px: u32,
) -> PyResult<Py<PyAny>> {
    let start_time = Instant::now();
    let image = plot_mel_spec(mel_spec, cmap, width_px, height_px);
    println!("plot_mel_spec_py plotting time: {:?}", start_time.elapsed());

    let start_time_encoding = Instant::now();
    let result = Python::with_gil(|py| {
        // Convert image to PNG bytes
        let mut buffer = Cursor::new(Vec::new());
        match image.write_to(&mut buffer, image::ImageFormat::Png) {
            Ok(_) => {
                // Convert to Python bytes object
                let bytes = PyBytes::new(py, &buffer.into_inner());
                Ok(bytes.into())
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to encode image: {}",
                e
            ))),
        }
    });
    println!(
        "plot_mel_spec_py encoding time: {:?}",
        start_time_encoding.elapsed()
    );
    println!("plot_mel_spec_py total time: {:?}", start_time.elapsed());
    result
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn create_mel_config(
    sample_rate: f32,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    top_db: f32,
    onesided: Option<bool>,
) -> MelConfig {
    let start_time = Instant::now();
    let spectrogram_config = SpectrogramConfig::new(onesided.unwrap_or(true));

    let result = MelConfig::new(
        sample_rate,
        n_fft,
        win_length,
        hop_length,
        f_min,
        f_max,
        n_mels,
        top_db,
        spectrogram_config,
    );
    println!(
        "create_mel_config execution time: {:?}",
        start_time.elapsed()
    );
    result
}

// Add implementation for MelConfig to convert from Python objects
#[cfg(feature = "python-bindings")]
impl<'py> FromPyObject<'py> for MelConfig {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let sample_rate: f32 = ob.getattr("sample_rate")?.extract()?;
        let n_fft: usize = ob.getattr("n_fft")?.extract()?;
        let win_length: usize = ob.getattr("win_length")?.extract()?;
        let hop_length: usize = ob.getattr("hop_length")?.extract()?;
        let f_min: f32 = ob.getattr("f_min")?.extract()?;
        let f_max: f32 = ob.getattr("f_max")?.extract()?;
        let n_mels: usize = ob.getattr("n_mels")?.extract()?;
        let top_db: f32 = ob.getattr("top_db")?.extract()?;

        // Get onesided flag or use default
        let onesided = match ob.getattr("onesided") {
            Ok(val) => val.extract()?,
            Err(_) => true, // Default value
        };

        let spectrogram_config = SpectrogramConfig::new(onesided);

        Ok(MelConfig::new(
            sample_rate,
            n_fft,
            win_length,
            hop_length,
            f_min,
            f_max,
            n_mels,
            top_db,
            spectrogram_config,
        ))
    }
}

// Add implementation for MelConfig to convert to Python objects
#[cfg(feature = "python-bindings")]
impl<'py> IntoPyObject<'py> for MelConfig {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("sample_rate", self.sample_rate)?;
        dict.set_item("n_fft", self.n_fft)?;
        dict.set_item("win_length", self.win_length)?;
        dict.set_item("hop_length", self.hop_length)?;
        dict.set_item("f_min", self.f_min)?;
        dict.set_item("f_max", self.f_max)?;
        dict.set_item("n_mels", self.n_mels)?;
        dict.set_item("top_db", self.top_db)?;
        dict.set_item("onesided", self.spectrogram_config.onesided)?;

        let dict_any = dict.into_pyobject(py)?.into_any();
        Ok(dict_any)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::complex::Complex;

    #[test]
    fn test_fft_constant() {
        let input = vec![1.0, 0.0, 0.0, 0.0]; // Changed to real input
        let output = fft(input, 4);
        _assert_complex_eq(output[0], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[2], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[3], Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_fft_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // Changed to real input
        let output = fft(input, 4);
        _assert_complex_eq(output[0], Complex::new(10.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(-2.0, 2.0));
        _assert_complex_eq(output[2], Complex::new(-2.0, 0.0));
        _assert_complex_eq(output[3], Complex::new(-2.0, -2.0));
    }

    #[test]
    fn test_fft_with_length_eight() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]; // Changed to real input
        let output = fft(input, 8);
        _assert_complex_eq(output[0], Complex::new(10.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(-0.41421356, -7.24264069));
        _assert_complex_eq(output[2], Complex::new(-2.0, 2.0));
        _assert_complex_eq(output[3], Complex::new(2.41421356, -1.24264069));
        _assert_complex_eq(output[4], Complex::new(-2.0, 0.0));
        _assert_complex_eq(output[5], Complex::new(2.41421356, 1.24264069));
        _assert_complex_eq(output[6], Complex::new(-2.0, -2.0));
        _assert_complex_eq(output[7], Complex::new(-0.41421356, 7.24264069));
    }
}
