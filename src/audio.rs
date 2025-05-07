use hound::WavReader;
use memmap2::Mmap;

// Only include this when the python-bindings feature is enabled
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

use rayon::prelude::*;
use std::io::{Seek, SeekFrom};

// Error type for WAV reading operations
#[derive(Debug)]
enum WavError {
    Io(()),
    Format(()),
}

impl From<std::io::Error> for WavError {
    fn from(_err: std::io::Error) -> Self {
        WavError::Io(())
    }
}

impl From<hound::Error> for WavError {
    fn from(err: hound::Error) -> Self {
        match err {
            hound::Error::IoError(_) => WavError::Io(()),
            _ => WavError::Format(()),
        }
    }
}

/// Read samples from WAV data based on format
fn read_samples(
    data: &[u8],
    spec: &hound::WavSpec,
    num_channels: usize,
) -> Result<Vec<Vec<f32>>, WavError> {
    let channel_samples: Vec<Vec<f32>> = match spec.sample_format {
        hound::SampleFormat::Float => {
            if spec.bits_per_sample == 32 {
                let frame_size = 4 * num_channels;
                (0..num_channels)
                    .map(|c| {
                        data.par_chunks_exact(frame_size)
                            .map(|chunk| {
                                let start = c * 4;
                                let mut arr = [0u8; 4];
                                arr.copy_from_slice(&chunk[start..start + 4]);
                                f32::from_le_bytes(arr)
                            })
                            .collect()
                    })
                    .collect()
            } else {
                return Err(WavError::Format(()));
            }
        }
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => {
                let frame_size = 2 * num_channels;
                let scale = 1.0 / i16::MAX as f32;
                (0..num_channels)
                    .map(|c| {
                        data.par_chunks_exact(frame_size)
                            .map(|chunk| {
                                let start = c * 2;
                                let mut arr = [0u8; 2];
                                arr.copy_from_slice(&chunk[start..start + 2]);
                                i16::from_le_bytes(arr) as f32 * scale
                            })
                            .collect()
                    })
                    .collect()
            }
            24 => {
                let frame_size = 3 * num_channels;
                let scale = 1.0 / 8_388_607.0;
                (0..num_channels)
                    .map(|c| {
                        data.par_chunks_exact(frame_size)
                            .map(|chunk| {
                                let start = c * 3;
                                let sample = ((chunk[start] as i32)
                                    | ((chunk[start + 1] as i32) << 8)
                                    | ((chunk[start + 2] as i32) << 16))
                                    as f32
                                    * scale;
                                sample
                            })
                            .collect()
                    })
                    .collect()
            }
            8 => {
                let frame_size = num_channels;
                let scale = 1.0 / 127.0;
                (0..num_channels)
                    .map(|c| {
                        data.par_chunks_exact(frame_size)
                            .map(|chunk| (chunk[c] as i8) as f32 * scale)
                            .collect()
                    })
                    .collect()
            }
            _ => {
                return Err(WavError::Format(()));
            }
        },
    };

    Ok(channel_samples)
}

pub fn generate_sine_wave(
    frequency: f32,
    duration: u32,
    sample_rate: u32,
    channels: u16,
) -> Vec<f32> {
    let mut audio_data = Vec::new();
    let samples_per_channel = sample_rate * duration;

    // Generate the base sine wave for one channel
    let single_channel: Vec<f32> = (0..samples_per_channel)
        .map(|x| x as f32 / sample_rate as f32)
        .map(|t| {
            let sample = (t * frequency * 2.0 * std::f32::consts::PI).sin();
            sample * i16::MAX as f32
        })
        .collect();

    // Duplicate the sine wave for each channel
    for i in 0..samples_per_channel {
        for _ in 0..channels {
            audio_data.push(single_channel[i as usize]);
        }
    }
    audio_data
}

fn _read_wav(path: &str) -> Result<(Vec<Vec<f32>>, usize), WavError> {
    let wav_reader = WavReader::open(path)?;

    let spec = wav_reader.spec();
    let num_channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as usize;
    if num_channels > 2 {
        return Err(WavError::Format(()));
    }

    let num_samples = wav_reader.len() as usize;

    let sample_bytes = (spec.bits_per_sample as usize + 7) / 8;
    let data_bytes = num_samples * sample_bytes;

    let mut buf_reader = wav_reader.into_inner(); // this is BufReader<File>
    let data_offset = buf_reader.seek(SeekFrom::Current(0))?;
    let file = buf_reader.into_inner();

    let mmap = unsafe { Mmap::map(&file)? };
    let data = &mmap[data_offset as usize..data_offset as usize + data_bytes];

    Ok((read_samples(data, &spec, num_channels)?, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_mono_wav() {
        let path = "./testdata/test.wav";
        let (channels, sample_rate) = _read_wav(path).unwrap();
        assert_eq!(channels.len(), 1); // Mono file
        assert_eq!(channels[0].len(), 16000);
        assert_eq!(sample_rate, 16000);
    }

    #[test]
    fn test_read_stereo_wav() {
        // Assumes you have a stereo test file
        let path = "./testdata/stereo_test.wav";
        if let Ok((channels, _sample_rate)) = _read_wav(path) {
            assert_eq!(channels.len(), 2); // Stereo file
            assert_eq!(channels[0].len(), channels[1].len());
        }
    }
}

/// Read a WAV file and return its data as a float array
///
/// Args:
///     path: Path to the WAV file
///     normalize: Whether to normalize audio to range [-1.0, 1.0] (default: true)
///
/// Returns:
///     tuple: (audio_data, sample_rate)
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn read_wav(path: String, normalize: Option<bool>) -> PyResult<(Vec<f32>, u32)> {
    let result = _read_wav(&path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read WAV file: {:?}", e))
    })?;

    let (channels, sample_rate) = result;

    // We'll just use the first channel for now
    let mut audio_data = if !channels.is_empty() {
        channels[0].clone()
    } else {
        Vec::new()
    };

    // Normalize if requested (default: true)
    if normalize.unwrap_or(true) && !audio_data.is_empty() {
        let max = audio_data
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, |a, b| a.max(b));

        if max > 0.0 {
            for sample in &mut audio_data {
                *sample /= max;
            }
        }
    }

    Ok((audio_data, sample_rate as u32))
}
