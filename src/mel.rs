use crate::colors;
use once_cell::sync::Lazy;
use rayon::prelude::*;
static GPU_DEVICE: Lazy<WgpuDevice> = Lazy::new(|| Default::default());

use crate::fft::fft;
use cubecl::wgpu::WgpuDevice;
use image::ImageBuffer;
use image::Rgb;
use log::{debug, error, info, warn};
use num::complex::Complex;
use std::default::Default;
use std::time::Instant;

// Only include this when the python-bindings feature is enabled
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
use pyo3::types::PyBytes;

use crate::audio::read_audio_file;

use std::io::Cursor;

// For faster PNG encoding

#[cfg(feature = "python-bindings")]
use image::ColorType;
type Runtime = cubecl::wgpu::WgpuRuntime;
/// Mel spectrogram from path
///
/// # Arguments
///
/// * `path` - The path to the audio file
/// * `n_fft` - The number of FFT bins
/// * `win_length` - The length of the window
/// * `hop_length` - The hop length
/// * `f_min` - The minimum frequency
/// * `f_max` - The maximum frequency
/// * `n_mels` - The number of mel bands
/// * `colormap` - The colormap to use
/// * `width_px` - The width of the image
/// * `height_px` - The height of the image
/// * `top_db` - The top dB value
/// # Returns
///
/// A PyBytes object containing the PNG image
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
    chunk_size: usize,
) -> PyResult<Py<PyAny>> {
    let start_time = Instant::now();
    let cmap = colors::Colormap::from_name(&colormap).ok_or_else(|| {
        warn!("Invalid colormap name: {}", colormap);
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid colormap name: {}",
            colormap
        ))
    })?;

    let (audio, sr) = read_audio_file(path, Some(true)).map_err(|e| {
        error!("Failed to read audio file: {}", e);
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read audio file: {}", e))
    })?;

    info!("Audio read time: {:?}", start_time.elapsed());
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
            chunk_size,
        ),
        audio,
    );
    info!(
        "Mel spectrogram generation time: {:?}",
        start_time.elapsed()
    );

    let start_time = Instant::now();
    let image = plot_mel_spec(mel_spec, cmap, width_px, height_px);
    info!("Plotting time: {:?}", start_time.elapsed());

    let start_time = Instant::now();
    let result = Python::with_gil(|py| {
        let mut buffer = Cursor::new(Vec::new());
        match image.write_to(&mut buffer, image::ImageFormat::Png) {
            Ok(_) => {
                let bytes = PyBytes::new(py, &buffer.into_inner());
                Ok(bytes.into())
            }
            Err(e) => {
                error!("Failed to encode image: {}", e);
                Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to encode image: {}",
                    e
                )))
            }
        }
    });
    info!("Image encoding time: {:?}", start_time.elapsed());
    result
}
fn gpu_spectrogram(
    waveform: Vec<f32>,
    n_fft: usize,
    _win_length: usize,
    _hop_length: usize,
    chunk_size: usize,
    onesided: bool,
) -> Vec<Vec<f32>> {
    let start_time = Instant::now();
    let device = GPU_DEVICE.clone();

    let batched_input: Vec<Vec<f32>> = waveform
        .chunks(chunk_size)
        .map(|chunk| {
            let mut frame = vec![0.0f32; chunk_size];
            frame[..chunk.len()].copy_from_slice(chunk);
            frame
        })
        .collect();
    debug!("Frame batching time: {:?}", start_time.elapsed());

    let fft_start = Instant::now();
    let half = if onesided { n_fft / 2 + 1 } else { n_fft };

    let spec: Vec<Vec<f32>> = batched_input
        .into_par_iter()
        .flat_map(|chunk| {
            chunk
                .chunks(n_fft)
                .map(|frame| {
                    let mut padded_frame = vec![0.0f32; n_fft];
                    padded_frame[..frame.len()].copy_from_slice(frame);
                    let (real, imag) = fft::<Runtime>(&device, padded_frame);
                    real.iter()
                        .zip(imag.iter())
                        .take(half)
                        .map(|(r, i)| (r.powi(2) + i.powi(2)).sqrt())
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<Vec<f32>>>()
        })
        .collect();

    let fft_time = fft_start.elapsed();
    info!("GPU spectrogram total time: {:?}", start_time.elapsed());
    debug!("FFT compute time: {:?}", fft_time);

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
impl Default for MelConfig {
    fn default() -> Self {
        // These values mirror the CLI defaults:
        let sample_rate = 22050.0_f32;
        let n_fft = 2048;
        let win_length = 2048;
        let hop_length = 512;
        let f_min = 0.0_f32;
        let f_max = 8000.0_f32;
        let n_mels = 128;
        let top_db = 80.0_f32;
        let spectrogram_config = SpectrogramConfig::default();
        let chunk_size = 4096_usize;
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
            chunk_size,
        }
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
    chunk_size: usize,
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
        chunk_size: usize,
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
            chunk_size,
        }
    }
}
pub fn mel_spectrogram_db(config: &MelConfig, waveform: Vec<f32>) -> Vec<Vec<f32>> {
    let top_db = config.top_db;
    let mel_spec: Vec<Vec<f32>> = mel_spectrogram(config, waveform);
    amplitude_to_db(mel_spec, top_db)
}

fn mel_spectrogram(config: &MelConfig, waveform: Vec<f32>) -> Vec<Vec<f32>> {
    let spectrogram = spectrogram(
        waveform,
        config.n_fft,
        config.win_length,
        config.hop_length,
        config.chunk_size,
        config.spectrogram_config.onesided,
    );
    let n_freqs = spectrogram[0].len();
    let fbanks = mel_filter_bank(
        n_freqs,
        config.f_min,
        config.f_max,
        config.n_mels,
        config.sample_rate,
    );
    spectrogram
        .into_par_iter()
        .map(|spec_row| {
            (0..config.n_mels)
                .map(|j| {
                    spec_row
                        .iter()
                        .zip(fbanks.iter())
                        .map(|(&s, fbank)| s * fbank[j])
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn spectrogram(
    waveform: Vec<f32>,
    n_fft: usize,
    _win_length: usize,
    _hop_length: usize,
    chunk_size: usize,
    onesided: bool,
) -> Vec<Vec<f32>> {
    let spec = gpu_spectrogram(
        waveform,
        n_fft,
        _win_length,
        _hop_length,
        chunk_size,
        onesided,
    );
    spec
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
            let norm = (val - smin) / (smax - smin + f32::EPSILON);
            let idx = (norm * 255.0).round().clamp(0.0, 255.0) as usize;
            let color = color_map[idx];
            image.put_pixel(px, py, Rgb(color));
        }
    }

    image
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

/// Generate a mel spectrogram
///
/// # Arguments
///
/// * `config` - A MelConfig object @see create_mel_config
/// * `waveform` - The waveform to generate the mel spectrogram from
/// # Returns
///
/// A computed mel spectrogram as a list of lists of floats
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn mel_spectrogram_db_py(config: MelConfig, waveform: Vec<f32>) -> Vec<Vec<f32>> {
    let start_time = Instant::now();
    let result = mel_spectrogram_db(&config, waveform);
    info!(
        "mel_spectrogram_db_py execution time: {:?}",
        start_time.elapsed()
    );
    result
}

/// Plot a mel spectrogram
///
/// # Arguments
///
/// * `mel_spec` - The mel spectrogram to plot
/// * `cmap` - The colormap to use
/// * `width_px` - The width of the image
/// * `height_px` - The height of the image
/// # Returns
///
/// A PyBytes object containing the PNG image
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn plot_mel_spec_py(
    mel_spec: Vec<Vec<f32>>,
    cmap: colors::Colormap,
    width_px: u32,
    height_px: u32,
) -> PyResult<Py<PyAny>> {
    use image::codecs::png::{CompressionType, FilterType, PngEncoder};
    use image::ImageEncoder;

    info!(
        "Starting plot_mel_spec_py with dimensions {}x{}",
        width_px, height_px
    );
    let start_time = Instant::now();

    let image = plot_mel_spec(mel_spec, cmap, width_px, height_px);
    let plotting_time = start_time.elapsed();
    info!("plot_mel_spec_py plotting time: {:?}", plotting_time);

    let start_time_encoding = Instant::now();
    let result = Python::with_gil(|py| {
        debug!(
            "Starting PNG encoding with buffer size: {} bytes",
            (width_px * height_px * 3) as usize
        );

        let start_time_buffer = Instant::now();
        let mut buffer = Vec::with_capacity((width_px * height_px * 3) as usize);
        let buffer_time = start_time_buffer.elapsed();
        debug!("Buffer allocation time: {:?}", buffer_time);

        let encoder =
            PngEncoder::new_with_quality(&mut buffer, CompressionType::Fast, FilterType::NoFilter);

        match encoder.write_image(&image, image.width(), image.height(), ColorType::Rgb8) {
            Ok(_) => {
                info!(
                    "PNG encoding successful, output size: {} bytes",
                    buffer.len()
                );
                let bytes = PyBytes::new(py, &buffer);
                Ok(bytes.into())
            }
            Err(e) => {
                error!("Failed to encode PNG: {}", e);
                Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to encode image: {}",
                    e
                )))
            }
        }
    });
    let encoding_time = start_time_encoding.elapsed();
    info!("plot_mel_spec_py encoding time: {:?}", encoding_time);
    info!("plot_mel_spec_py total time: {:?}", start_time.elapsed());
    debug!("plot_mel_spec_py performance breakdown:");
    debug!("  - Plotting: {:?}", plotting_time);
    debug!("  - Encoding: {:?}", encoding_time);
    result
}

/// Create a MelConfig object
///
/// # Arguments
///
/// * `sample_rate` - The sample rate of the audio
/// * `n_fft` - The number of FFT bins
/// * `win_length` - The length of the window
/// * `hop_length` - The hop length
/// * `f_min` - The minimum frequency
/// * `f_max` - The maximum frequency
/// * `n_mels` - The number of mel bands
/// * `top_db` - The top dB value
/// * `onesided` - Whether to use a one-sided FFT
/// # Returns
///
/// A MelConfig object for use in the mel_spectrogram_db_py function
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
    chunk_size: usize,
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
        chunk_size,
    );
    debug!(
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
        let chunk_size: usize = ob.getattr("chunk_size")?.extract()?;
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
            chunk_size,
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
