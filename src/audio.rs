use memmap2::Mmap;
use std::fs::File;
use std::io::{Cursor, Read};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// Only include this when the python-bindings feature is enabled
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

use log::{debug, info};
use std::time::Instant;

// Error type for audio reading operations
#[derive(Debug)]
pub enum AudioError {
    Io(std::io::Error),
    Decode(SymphoniaError),
    UnsupportedFormat,
}

impl From<std::io::Error> for AudioError {
    fn from(err: std::io::Error) -> Self {
        AudioError::Io(err)
    }
}

impl From<SymphoniaError> for AudioError {
    fn from(err: SymphoniaError) -> Self {
        AudioError::Decode(err)
    }
}

/// Read audio samples from a file using Symphonia
fn read_audio(
    path: &str,
    offset: Option<f32>,
    duration: Option<f32>,
) -> Result<(Vec<Vec<f32>>, usize), AudioError> {
    let func_start = Instant::now();
    info!("read_audio: starting for path {}", path);
    let mut file = File::open(path)?;
    let mss = {
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let cursor = Cursor::new(buffer);
        MediaSourceStream::new(Box::new(cursor), Default::default())
    };

    // Create a hint to help the format registry guess the format
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension() {
        if let Some(ext_str) = ext.to_str() {
            hint.with_extension(ext_str);
        }
    }

    // Probe the format
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let probe_start = Instant::now();
    let mut probed =
        symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
    info!("read_audio: format probed in {:?}", probe_start.elapsed());

    // Get the default track
    let track = probed
        .format
        .default_track()
        .ok_or(AudioError::UnsupportedFormat)?;
    let track_id = track.id;

    // Create a decoder for the track
    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

    // Get the sample rate and number of channels
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100) as usize;
    let num_channels = track
        .codec_params
        .channels
        .unwrap_or(
            symphonia::core::audio::Channels::FRONT_LEFT
                | symphonia::core::audio::Channels::FRONT_RIGHT,
        )
        .count() as usize;

    // Calculate start and end samples based on offset and duration
    let start_sample = if let Some(offset) = offset {
        (offset * sample_rate as f32) as u64
    } else {
        0
    };

    let end_sample = if let Some(duration) = duration {
        start_sample + (duration * sample_rate as f32) as u64
    } else {
        u64::MAX
    };

    // Initialize channel buffers with capacity reservation
    let estimate_samples = duration.map_or(0.0, |d| d * sample_rate as f32) as usize;
    let mut channel_buffers = Vec::with_capacity(num_channels);
    for _ in 0..num_channels {
        let mut buf = Vec::new();
        buf.reserve(estimate_samples);
        channel_buffers.push(buf);
    }
    let mut current_sample = 0u64;
    let mut sample_buf = None;

    // Decode the audio data
    let decode_start = Instant::now();
    while let Ok(packet) = probed.format.next_packet() {
        let packet_start = Instant::now();
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                match audio_buf {
                    AudioBufferRef::F32(buf_ref) => {
                        let channels = buf_ref.spec().channels.count();
                        for ch in 0..channels {
                            let slice = buf_ref.chan(ch);
                            channel_buffers[ch].extend_from_slice(slice);
                        }
                    }
                    _ => {
                        // Existing interleaved processing
                        if sample_buf.is_none() {
                            let spec = *audio_buf.spec();
                            let duration = audio_buf.capacity() as u64;
                            sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                        }
                        if let Some(buf) = &mut sample_buf {
                            buf.copy_interleaved_ref(audio_buf);
                            let samples = buf.samples();
                            for chunk in samples.chunks(num_channels) {
                                if current_sample >= start_sample {
                                    for (channel, &sample) in chunk.iter().enumerate() {
                                        channel_buffers[channel].push(sample);
                                    }
                                }
                                current_sample += 1;
                                if current_sample >= end_sample {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(AudioError::Decode(e)),
        }

        if current_sample >= end_sample {
            break;
        }
    }

    info!(
        "read_audio: decoding completed in {:?}",
        decode_start.elapsed()
    );
    info!("read_audio: total time {:?}", func_start.elapsed());
    Ok((channel_buffers, sample_rate))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_audio() {
        let path = "./testdata/test.wav";
        let (channels, sample_rate) = read_audio(path, None, None).unwrap();
        assert_eq!(channels.len(), 1); // Mono file
        assert_eq!(sample_rate, 16000);
    }
}

/// Read an audio file and return its data as a float array
///
/// Args:
///     path: Path to the audio file
///     normalize: Whether to normalize audio to range [-1.0, 1.0] (default: true)
///     offset: Offset in seconds from the start of the file (default: None)
///     duration: Duration in seconds to read from the file (default: None)
///
/// Returns:
///     tuple: (audio_data, sample_rate)
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn read_audio_file(path: String, normalize: Option<bool>) -> PyResult<(Vec<f32>, u32)> {
    let result = read_audio(&path, None, None).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read audio file: {:?}", e))
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
