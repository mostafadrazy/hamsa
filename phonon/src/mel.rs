//! Log-mel features matching transformers' ParakeetFeatureExtractor:
//! pre-emphasis 0.97 → STFT (n_fft 512, hop 160, 256 zeros of padding each side, symmetric Hann of 400 centred in 512)
//! → power → slaney mel (128 bins, 0–8 kHz) → ln(x + 2⁻²⁴) → per-feature normalisation over the frames
//! ((x − mean) / (std + 1e-5), unbiased variance).

use std::f64::consts::PI;
use std::sync::OnceLock;

use rayon::prelude::*;
use realfft::RealFftPlanner;

pub const N_MELS: usize = 128;
const N_FFT: usize = 512;
const N_FREQS: usize = N_FFT / 2 + 1;
const WIN: usize = 400;
const HOP: usize = 160;
const SAMPLE_RATE: f64 = 16_000.0;
const PREEMPHASIS: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24
const NORM_EPS: f32 = 1e-5;

fn hz_to_mel(hz: f64) -> f64 {
    if hz < 1000.0 {
        hz * 3.0 / 200.0
    } else {
        15.0 + 27.0 * (hz / 1000.0).ln() / 6.4f64.ln()
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15.0 {
        mel * 200.0 / 3.0
    } else {
        1000.0 * ((mel - 15.0) * 6.4f64.ln() / 27.0).exp()
    }
}

/// [N_MELS × N_FREQS], librosa's slaney-normalised filterbank.
fn filterbank() -> &'static [f32] {
    static FB: OnceLock<Vec<f32>> = OnceLock::new();
    FB.get_or_init(|| {
        let (mel_lo, mel_hi) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2.0));
        let points: Vec<f64> = (0..N_MELS + 2)
            .map(|i| mel_to_hz(mel_lo + (mel_hi - mel_lo) * i as f64 / (N_MELS + 1) as f64))
            .collect();
        let mut fb = vec![0.0f32; N_MELS * N_FREQS];
        for m in 0..N_MELS {
            let (lo, center, hi) = (points[m], points[m + 1], points[m + 2]);
            let norm = 2.0 / (hi - lo);
            for f in 0..N_FREQS {
                let hz = SAMPLE_RATE * f as f64 / N_FFT as f64;
                let v = if hz >= lo && hz <= center {
                    (hz - lo) / (center - lo)
                } else if hz > center && hz <= hi {
                    (hi - hz) / (hi - center)
                } else {
                    0.0
                };
                fb[m * N_FREQS + f] = (v * norm) as f32;
            }
        }
        fb
    })
}

/// Returns ([frames × 128] features, frames), or None when the audio is under two frames long.
pub fn log_mel(audio: &[f32]) -> Option<(Vec<f32>, usize)> {
    let n = audio.len();
    let frames = n / HOP;
    if frames < 2 {
        return None;
    }

    let pad = N_FFT / 2;
    let mut padded = vec![0.0f32; n + 2 * pad];
    padded[pad] = audio[0];
    for i in 1..n {
        padded[pad + i] = audio[i] - PREEMPHASIS * audio[i - 1];
    }

    let mut window = [0.0f32; N_FFT];
    let off = (N_FFT - WIN) / 2;
    for i in 0..WIN {
        window[off + i] = (0.5 - 0.5 * (2.0 * PI * i as f64 / (WIN - 1) as f64).cos()) as f32;
    }

    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let fb = filterbank();
    let mut mel = vec![0.0f32; frames * N_MELS];
    mel.par_chunks_mut(N_MELS).enumerate().for_each_init(
        || (fft.make_input_vec(), fft.make_output_vec(), vec![0.0f32; N_FREQS]),
        |(input, spectrum, power), (f, out)| {
            let frame = &padded[f * HOP..f * HOP + N_FFT];
            for i in 0..N_FFT {
                input[i] = frame[i] * window[i];
            }
            fft.process(input, spectrum).expect("fft buffer sizes");
            for k in 0..N_FREQS {
                power[k] = spectrum[k].re * spectrum[k].re + spectrum[k].im * spectrum[k].im;
            }
            for m in 0..N_MELS {
                let row = &fb[m * N_FREQS..(m + 1) * N_FREQS];
                let v: f32 = row.iter().zip(power.iter()).map(|(a, b)| a * b).sum();
                out[m] = (v + LOG_GUARD).ln();
            }
        },
    );

    // Per-feature normalisation over the frames.
    let mut mean = [0.0f64; N_MELS];
    let mut var = [0.0f64; N_MELS];
    for row in mel.chunks_exact(N_MELS) {
        for m in 0..N_MELS {
            mean[m] += row[m] as f64;
        }
    }
    for v in mean.iter_mut() {
        *v /= frames as f64;
    }
    for row in mel.chunks_exact(N_MELS) {
        for m in 0..N_MELS {
            let d = row[m] as f64 - mean[m];
            var[m] += d * d;
        }
    }
    let mut mu = [0.0f32; N_MELS];
    let mut inv = [0.0f32; N_MELS];
    for m in 0..N_MELS {
        let std = (var[m] / (frames - 1) as f64).sqrt() as f32;
        inv[m] = 1.0 / (std + NORM_EPS);
        mu[m] = mean[m] as f32;
    }
    for row in mel.chunks_exact_mut(N_MELS) {
        for m in 0..N_MELS {
            row[m] = (row[m] - mu[m]) * inv[m];
        }
    }
    Some((mel, frames))
}
