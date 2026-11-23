//! Microphone capture with cpal (WASAPI on Windows).
//!
//! A `Recording` owns a capture thread: the cpal stream isn't `Send`, so it is
//! created, played and dropped on that thread. Samples are mixed to mono at the
//! device's native rate; `to_16k` converts them for the speech model.

use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, SizedSample, Stream, StreamConfig};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub const TARGET_RATE: u32 = 16_000;

pub struct Captured {
    /// Mono samples at `sample_rate`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub struct Recording {
    stop_tx: Sender<()>,
    done_rx: Receiver<Captured>,
    level: Arc<AtomicU32>,
    /// Filled by the capture thread; read for the live preview.
    buffer: Arc<Mutex<Vec<f32>>>,
    rate: Arc<AtomicU32>,
}

impl Recording {
    /// RMS of the most recent audio buffer, 0.0–1.0.
    pub fn rms(&self) -> f32 {
        f32::from_bits(self.level.load(Relaxed))
    }

    /// A copy of the last `max_seconds` recorded so far (for the live preview), and how many seconds that skips.
    pub fn snapshot(&self, max_seconds: f32) -> Option<(Captured, f32)> {
        let rate = self.rate.load(Relaxed);
        if rate == 0 {
            return None;
        }
        let b = self.buffer.lock().ok()?;
        let keep = ((max_seconds * rate as f32) as usize).min(b.len());
        let skipped = (b.len() - keep) as f32 / rate as f32;
        Some((Captured { samples: b[b.len() - keep..].to_vec(), sample_rate: rate }, skipped))
    }

    /// Stop the stream and return everything recorded.
    pub fn finish(self) -> Result<Captured, String> {
        let _ = self.stop_tx.send(());
        self.done_rx
            .recv()
            .map_err(|_| "The microphone stopped unexpectedly.".to_string())
    }
}

/// Names of all input devices, plus the Windows default.
pub fn list_inputs() -> (Vec<String>, Option<String>) {
    let host = cpal::default_host();
    let default = host.default_input_device().and_then(|d| d.name().ok());
    let devices = host
        .input_devices()
        .map(|it| it.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default();
    (devices, default)
}

fn pick_device(host: &Host, name: Option<&str>) -> Result<Device, String> {
    if let Some(wanted) = name {
        if let Ok(mut devices) = host.input_devices() {
            if let Some(d) = devices.find(|d| d.name().map(|n| n == wanted).unwrap_or(false)) {
                return Ok(d);
            }
        }
    }
    host.default_input_device()
        .ok_or_else(|| "No microphone found. Plug one in or check Windows sound settings.".into())
}

fn build_stream<T>(
    device: &Device,
    config: &StreamConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
    level: Arc<AtomicU32>,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = config.channels.max(1) as usize;
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let mut sum_sq = 0.0f32;
            let mut frames = 0usize;
            if let Ok(mut out) = buffer.lock() {
                for frame in data.chunks(channels) {
                    let mono = frame.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / channels as f32;
                    sum_sq += mono * mono;
                    frames += 1;
                    out.push(mono);
                }
            }
            let rms = (sum_sq / frames.max(1) as f32).sqrt();
            level.store(rms.to_bits(), Relaxed);
        },
        |err| eprintln!("hamsa: audio stream error: {err}"),
        None,
    )
}

/// Open the microphone (by name, or the Windows default) and start recording.
pub fn start(device_name: Option<String>) -> Result<Recording, String> {
    let (stop_tx, stop_rx) = channel::<()>();
    let (done_tx, done_rx) = channel::<Captured>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();
    let level = Arc::new(AtomicU32::new(0));
    let thread_level = level.clone();
    let shared = Arc::new(Mutex::new(Vec::new()));
    let thread_shared = shared.clone();
    let rate_cell = Arc::new(AtomicU32::new(0));
    let thread_rate = rate_cell.clone();

    std::thread::Builder::new()
        .name("hamsa-capture".into())
        .spawn(move || {
            let opened = (|| -> Result<(Stream, Arc<Mutex<Vec<f32>>>, u32), String> {
                let host = cpal::default_host();
                let device = pick_device(&host, device_name.as_deref())?;
                let supported = device
                    .default_input_config()
                    .map_err(|e| format!("Can't open the microphone: {e}"))?;
                let rate = supported.sample_rate().0;
                let format = supported.sample_format();
                let config: StreamConfig = supported.into();
                let buffer = thread_shared;
                if let Ok(mut b) = buffer.lock() {
                    b.reserve(rate as usize * 30);
                }
                thread_rate.store(rate, Relaxed);
                let b = buffer.clone();
                let l = thread_level.clone();
                let stream = match format {
                    SampleFormat::F32 => build_stream::<f32>(&device, &config, b, l),
                    SampleFormat::I16 => build_stream::<i16>(&device, &config, b, l),
                    SampleFormat::U16 => build_stream::<u16>(&device, &config, b, l),
                    SampleFormat::I32 => build_stream::<i32>(&device, &config, b, l),
                    other => return Err(format!("Unsupported microphone format: {other:?}")),
                }
                .map_err(|e| format!("Can't open the microphone: {e}"))?;
                stream.play().map_err(|e| format!("Can't start the microphone: {e}"))?;
                Ok((stream, buffer, rate))
            })();

            match opened {
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
                Ok((stream, buffer, rate)) => {
                    let _ = ready_tx.send(Ok(()));
                    let _ = stop_rx.recv();
                    drop(stream);
                    let samples = buffer.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default();
                    let _ = done_tx.send(Captured { samples, sample_rate: rate });
                }
            }
        })
        .map_err(|e| e.to_string())?;

    ready_rx
        .recv()
        .map_err(|_| "The microphone thread failed to start.".to_string())??;
    Ok(Recording { stop_tx, done_rx, level, buffer: shared, rate: rate_cell })
}

/// Convert mono audio to 16 kHz with a windowed-sinc resampler.
pub fn to_16k(samples: &[f32], rate: u32) -> Result<Vec<f32>, String> {
    if rate == TARGET_RATE || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk = 1024;
    let ratio = TARGET_RATE as f64 / rate as f64;
    let mut resampler =
        SincFixedIn::<f32>::new(ratio, 1.0, params, chunk, 1).map_err(|e| e.to_string())?;
    let delay = resampler.output_delay();

    let mut out = Vec::with_capacity((samples.len() as f64 * ratio) as usize + chunk);
    let mut pos = 0;
    while pos + chunk <= samples.len() {
        let block = resampler
            .process(&[&samples[pos..pos + chunk]], None)
            .map_err(|e| e.to_string())?;
        out.extend_from_slice(&block[0]);
        pos += chunk;
    }
    if pos < samples.len() {
        let block = resampler
            .process_partial(Some(&[&samples[pos..]]), None)
            .map_err(|e| e.to_string())?;
        out.extend_from_slice(&block[0]);
    }
    let tail = resampler
        .process_partial::<&[f32]>(None, None)
        .map_err(|e| e.to_string())?;
    out.extend_from_slice(&tail[0]);

    let expected = (samples.len() as f64 * ratio).round() as usize;
    let start = delay.min(out.len());
    let end = (start + expected).min(out.len());
    Ok(out[start..end].to_vec())
}

/// Write 16 kHz mono audio as a 16-bit WAV file.
pub fn save_wav(path: &Path, pcm: &[f32]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
    for s in pcm {
        writer
            .write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resamples_48k_to_16k_with_correct_length_and_pitch() {
        // One second of a 440 Hz tone at 48 kHz.
        let rate = 48_000;
        let tone: Vec<f32> = (0..rate)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin() * 0.5)
            .collect();
        let out = to_16k(&tone, rate).unwrap();
        assert_eq!(out.len(), 16_000);
        // Count zero crossings in the middle: 440 Hz → ~880 per second.
        let mid = &out[2_000..14_000];
        let crossings = mid.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        let per_second = crossings as f32 / (mid.len() as f32 / 16_000.0);
        assert!((per_second - 880.0).abs() < 10.0, "got {per_second} crossings/s");
    }

    #[test]
    fn meter_maps_decibels() {
        assert_eq!(meter(0.0), 0.0);
        assert!(meter(0.001) < 0.05); // −60 dB: silent
        assert!(meter(0.3) > 0.95); // loud speech
    }

    /// Opens the real default microphone for half a second.
    /// Run with: cargo test -- --ignored opens_default_microphone
    #[test]
    #[ignore]
    fn opens_default_microphone() {
        let rec = start(None).expect("open microphone");
        std::thread::sleep(std::time::Duration::from_millis(500));
        let cap = rec.finish().expect("finish");
        assert!(cap.sample_rate >= 8_000);
        assert!(cap.samples.len() as u32 > cap.sample_rate / 4, "too few samples: {}", cap.samples.len());
    }
}

/// Map RMS to a 0–1 meter value on a decibel scale (−55 dB … −10 dB).
pub fn meter(rms: f32) -> f32 {
    if rms <= 0.0 {
        return 0.0;
    }
    let db = 20.0 * rms.log10();
    ((db + 55.0) / 45.0).clamp(0.0, 1.0)
}
