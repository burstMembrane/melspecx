use once_cell::sync::Lazy;
use std::sync::Once;

static INIT_LOGGER: Once = Once::new();
static LOGGER: Lazy<()> = Lazy::new(|| {
    INIT_LOGGER.call_once(|| {
        env_logger::init();
        log::info!("Initialized melspecx logger");
    });
});

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

pub mod audio;
pub mod colors;
pub mod fft;
pub mod mel;
/// MelSpecX: Fast Mel Spectrogram Generation
///
/// A Python module implemented in Rust for efficiently creating
/// mel spectrograms from audio files with GPU acceleration.
#[cfg(feature = "python-bindings")]
#[pymodule]
fn melspecx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logger when module is loaded
    let _ = *LOGGER;

    #[cfg(feature = "python-bindings")]
    {
        // add docstrings to the functions
        m.add_function(wrap_pyfunction!(audio::read_audio_file, m)?)?;
        m.add_function(wrap_pyfunction!(mel::mel_spec_from_path, m)?)?;
        m.add_function(wrap_pyfunction!(mel::plot_mel_spec_py, m)?)?;
        m.add_function(wrap_pyfunction!(mel::create_mel_config, m)?)?;
        m.add_function(wrap_pyfunction!(mel::mel_spectrogram_db_py, m)?)?;
    }
    Ok(())
}
