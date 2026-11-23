//! Phonon-2.5 speech recognition in pure Rust.
//!
//! Runs Fermion Research's Phonon-2.5 and Phonon-2.5 English (five-value students of NVIDIA's Parakeet-TDT 0.6B v3)
//! straight from their release archive. The encoder's five-value weights stay packed in memory (~225 MB);
//! everything else (mel front end, FastConformer encoder, LSTM + TDT decoder) is implemented here.
//!
//! The computation follows transformers' ParakeetForTDT; eschmidbauer/phonon-2.5-c (MIT) served as the reference
//! for the container layout and the decoding loop.

mod boost;
mod container;
mod decoder;
mod encoder;
mod gpu;
mod gpu_shaders;
#[cfg(target_arch = "x86_64")]
mod kernel;
mod linear;
mod mel;
mod profile;

/// Cumulative milliseconds per encoder stage since the last call (for profiling).
pub fn profile_report() -> Vec<(&'static str, f64)> {
    profile::report()
}

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

pub const SAMPLE_RATE: u32 = 16_000;
/// Audio longer than this is cut at pauses into windows; the model is trained on utterances up to ~35 s.
const WINDOW_MAX_S: usize = 35;
const WINDOW_MIN_S: usize = 25;

const SPECIAL_IDS: [u32; 3] = [0, 2, 3]; // <unk>, <pad>, <|endoftext|>

pub struct Transcript {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub timings: Timings,
}

/// Milliseconds spent in each stage, summed over windows.
#[derive(Default, Clone, Copy, Debug)]
pub struct Timings {
    pub mel_ms: f32,
    pub encoder_ms: f32,
    pub decoder_ms: f32,
}

/// An alphabet the output can be locked to. The models detect the language themselves and take no language
/// setting; a script lock only stops output in the wrong alphabet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Script {
    Latin,
    Cyrillic,
    Greek,
}

/// The script of one of Phonon-2.5's 25 languages (ISO 639-1 code), or None for unknown codes / "auto".
pub fn script_for_language(code: &str) -> Option<Script> {
    match code {
        "ru" | "uk" | "bg" => Some(Script::Cyrillic),
        "el" => Some(Script::Greek),
        "en" | "es" | "fr" | "de" | "it" | "pt" | "nl" | "pl" | "ro" | "cs" | "sk" | "hr" | "sv" | "da" | "fi"
        | "et" | "lv" | "lt" | "hu" | "sl" | "mt" => Some(Script::Latin),
        _ => None,
    }
}

/// The script of a vocabulary piece's first letter; None for pieces without letters (punctuation, digits).
fn piece_script(piece: &str) -> Option<Script> {
    for ch in piece.chars() {
        let c = ch as u32;
        if (0x0400..=0x052F).contains(&c) {
            return Some(Script::Cyrillic);
        }
        if (0x0370..=0x03FF).contains(&c) || (0x1F00..=0x1FFF).contains(&c) {
            return Some(Script::Greek);
        }
        if ch.is_ascii_alphabetic() || (0x00C0..=0x024F).contains(&c) || (0x1E00..=0x1EFF).contains(&c) {
            return Some(Script::Latin);
        }
    }
    None
}

pub struct Engine {
    encoder: encoder::Encoder,
    decoder: decoder::Decoder,
    vocab: Vec<String>,
    /// Per script: which tokens may be emitted (letters of that script, or no letters at all).
    script_masks: [Vec<bool>; 3],
    /// The encoder on the GPU, when one is usable.
    gpu: Option<gpu::Gpu>,
    gpu_error: Option<String>,
    /// Model folder, to reload the CPU encoder layers if the GPU fails after they were freed.
    dir: PathBuf,
    cpu_layers: OnceLock<Result<encoder::Encoder, String>>,
}

/// Read just the encoder from a model folder.
fn load_encoder(dir: &Path) -> Result<encoder::Encoder, String> {
    let container = dir.join("model.fermion");
    let bytes = fs::read(&container).map_err(|e| format!("can't read {}: {e}", container.display()))?;
    let mut records = container::read(&bytes)?;
    encoder::Encoder::load(&mut records)
}

pub use boost::Boost;

/// Per-call transcription options.
#[derive(Default, Clone, Copy)]
pub struct Options<'a> {
    /// Keep the output in one alphabet (multilingual model).
    pub script: Option<Script>,
    /// Favour these dictionary terms (from `Engine::boost`).
    pub boost: Option<&'a Boost>,
}

/// Where the encoder runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Compute {
    /// A dedicated graphics card when there is one, else the CPU.
    Auto,
    /// Any hardware GPU, built-in ones included; the CPU only if there is none.
    Gpu,
    Cpu,
}

impl Engine {
    /// Load from an unpacked model directory (`model.fermion` + `config.json`), using a GPU when there is one.
    pub fn load(dir: &Path) -> Result<Self, String> {
        Self::load_with(dir, Compute::Auto)
    }

    pub fn load_with(dir: &Path, compute: Compute) -> Result<Self, String> {
        let container = dir.join("model.fermion");
        let bytes = fs::read(&container).map_err(|e| format!("can't read {}: {e}", container.display()))?;
        let mut records = container::read(&bytes)?;
        drop(bytes);

        let mut encoder = encoder::Encoder::load(&mut records)?;
        let decoder = decoder::Decoder::load(&mut records)?;
        records.retain(|k, _| !k.ends_with("num_batches_tracked"));
        if !records.is_empty() {
            let mut left: Vec<_> = records.keys().cloned().collect();
            left.sort();
            return Err(format!("model has weights this engine doesn't use: {}", left.join(", ")));
        }

        let config_path = dir.join("config.json");
        let config: serde_json::Value = serde_json::from_slice(
            &fs::read(&config_path).map_err(|e| format!("can't read {}: {e}", config_path.display()))?,
        )
        .map_err(|e| format!("config.json: {e}"))?;
        let labels = config["labels"].as_array().ok_or("config.json has no `labels`")?;
        let mut vocab: Vec<String> = labels.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        if vocab.len() != decoder::BLANK {
            return Err(format!("vocabulary has {} entries, expected {}", vocab.len(), decoder::BLANK));
        }
        vocab.push(String::new()); // blank

        let mask = |s: Script| vocab.iter().map(|p| piece_script(p).map_or(true, |ps| ps == s)).collect::<Vec<bool>>();
        let script_masks = [mask(Script::Latin), mask(Script::Cyrillic), mask(Script::Greek)];

        let (gpu, gpu_error) = if compute == Compute::Cpu || std::env::var_os("PHONON_CPU").is_some() {
            (None, Some("CPU selected".to_string()))
        } else {
            match gpu::Gpu::new(&encoder, compute == Compute::Gpu) {
                Ok(g) => (Some(g), None),
                Err(e) => (None, Some(e)),
            }
        };
        // With the encoder on the GPU, its CPU copy is ~250 MB of RAM doing nothing: free it.
        if gpu.is_some() {
            encoder.drop_layers();
        }
        Ok(Self {
            encoder,
            decoder,
            vocab,
            script_masks,
            gpu,
            gpu_error,
            dir: dir.to_path_buf(),
            cpu_layers: OnceLock::new(),
        })
    }

    /// "GPU · <name>" or "CPU".
    pub fn device(&self) -> String {
        match &self.gpu {
            Some(g) => format!("GPU · {}", g.name),
            None => "CPU".to_string(),
        }
    }

    /// Why the GPU isn't used, when it isn't.
    pub fn gpu_unavailable_reason(&self) -> Option<&str> {
        self.gpu_error.as_deref()
    }

    /// Encoder on the GPU when possible; the CPU otherwise or if the GPU run fails.
    fn encode(&self, mel: &[f32], frames: usize) -> (Vec<f32>, usize) {
        let (x, t) = self.encoder.subsample(mel, frames);
        if let Some(g) = &self.gpu {
            if t <= gpu::MAX_FRAMES {
                match g.forward(&x, t) {
                    Ok(out) => return (out, t),
                    Err(e) => eprintln!("phonon: GPU encoder failed ({e}); using the CPU"),
                }
            }
        }
        let cpu = if self.encoder.has_layers() {
            &self.encoder
        } else {
            match self.cpu_layers.get_or_init(|| load_encoder(&self.dir)) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("phonon: couldn't reload the CPU encoder ({e})");
                    return (Vec::new(), 0);
                }
            }
        };
        (cpu.forward_blocks(x, t), t)
    }

    /// Bytes of weights held in memory.
    pub fn memory_bytes(&self) -> usize {
        self.encoder.bytes() + self.decoder.bytes()
    }

    /// Transcribe mono 16 kHz audio.
    pub fn transcribe(&self, pcm: &[f32]) -> Transcript {
        self.transcribe_with(pcm, None)
    }

    /// Prepare dictionary terms for `Options::boost`. Build once and reuse: it's cheap but not free.
    pub fn boost(&self, terms: &[String]) -> Boost {
        let ids: std::collections::HashMap<&str, u32> =
            self.vocab.iter().enumerate().filter(|(_, p)| !p.is_empty()).map(|(i, p)| (p.as_str(), i as u32)).collect();
        Boost::build(terms, &ids)
    }

    /// Transcribe, optionally locking the output to one alphabet.
    pub fn transcribe_with(&self, pcm: &[f32], script: Option<Script>) -> Transcript {
        self.transcribe_opts(pcm, &Options { script, boost: None })
    }

    pub fn transcribe_opts(&self, pcm: &[f32], opts: &Options) -> Transcript {
        let script = opts.script;
        let boost = opts.boost.filter(|b| !b.is_empty());
        let mask = script.map(|s| {
            self.script_masks[match s {
                Script::Latin => 0,
                Script::Cyrillic => 1,
                Script::Greek => 2,
            }]
            .as_slice()
        });
        let mut token_ids = Vec::new();
        let mut parts = Vec::new();
        let mut timings = Timings::default();
        for window in split_windows(pcm) {
            let ids = self.decode_window(window, &mut timings, mask, boost);
            let text = self.text(&ids);
            if !text.is_empty() {
                parts.push(text);
            }
            token_ids.extend(ids);
        }
        Transcript { text: parts.join(" "), token_ids, timings }
    }

    fn decode_window(&self, pcm: &[f32], timings: &mut Timings, mask: Option<&[bool]>, boost: Option<&Boost>) -> Vec<u32> {
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f32() * 1000.0;
        let t = std::time::Instant::now();
        let Some((mel, frames)) = mel::log_mel(pcm) else {
            return Vec::new();
        };
        timings.mel_ms += ms(t);
        let t = std::time::Instant::now();
        let (enc, frames_out) = self.encode(&mel, frames);
        timings.encoder_ms += ms(t);
        let t = std::time::Instant::now();
        let ids = self.decoder.greedy(&enc, frames_out, mask, boost).into_iter().map(|(id, _frame)| id).collect();
        timings.decoder_ms += ms(t);
        ids
    }

    /// SentencePiece pieces → text: ▁ becomes a space; special ids and blank are dropped.
    fn text(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if SPECIAL_IDS.contains(&id) || id as usize == decoder::BLANK {
                continue;
            }
            if let Some(piece) = self.vocab.get(id as usize) {
                out.push_str(&piece.replace('\u{2581}', " "));
            }
        }
        out.trim().to_string()
    }
}

/// Cut long audio into ≤ 35 s windows, each ending at the quietest 0.4 s between 25 and 35 s.
fn split_windows(pcm: &[f32]) -> Vec<&[f32]> {
    let sr = SAMPLE_RATE as usize;
    let mut windows = Vec::new();
    let mut start = 0;
    while pcm.len() - start > WINDOW_MAX_S * sr {
        let (lo, hi) = (start + WINDOW_MIN_S * sr, start + WINDOW_MAX_S * sr);
        let span = sr * 2 / 5;
        let step = sr / 100;
        let mut best = (f32::INFINITY, hi);
        let mut s = lo;
        while s + span <= hi {
            let energy: f32 = pcm[s..s + span].iter().map(|v| v * v).sum();
            if energy < best.0 {
                best = (energy, s + span / 2);
            }
            s += step;
        }
        windows.push(&pcm[start..best.1]);
        start = best.1;
    }
    windows.push(&pcm[start..]);
    windows
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Unpack a Phonon `.bps.tar.zst` release archive into `dest` and check the container's checksum.
/// Returns the directory holding `model.fermion`.
pub fn unpack(archive: &Path, dest: &Path) -> Result<PathBuf, String> {
    fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    let file = fs::File::open(archive).map_err(|e| format!("can't open {}: {e}", archive.display()))?;
    let zst = zstd::stream::read::Decoder::new(file).map_err(|e| e.to_string())?;
    tar::Archive::new(zst).unpack(dest).map_err(|e| format!("unpacking failed: {e}"))?;

    let dir = find_model_dir(dest).ok_or("the archive has no model.fermion")?;
    let manifest_path = dir.join("packed_manifest.json");
    if let Ok(raw) = fs::read(&manifest_path) {
        let manifest: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        if let Some(expected) = manifest["container_sha256"].as_str() {
            let actual = sha256_file(&dir.join("model.fermion"))?;
            if actual != expected {
                return Err(format!("model.fermion checksum mismatch: {actual} != {expected}"));
            }
        }
    }
    Ok(dir)
}

/// `dir` itself or a subdirectory (up to two levels) containing model.fermion.
pub fn find_model_dir(dir: &Path) -> Option<PathBuf> {
    if dir.join("model.fermion").is_file() {
        return Some(dir.to_path_buf());
    }
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if p.join("model.fermion").is_file() {
                return Some(p);
            }
            if let Some(found) = fs::read_dir(&p).ok().and_then(|rd| {
                rd.flatten().map(|e| e.path()).find(|q| q.is_dir() && q.join("model.fermion").is_file())
            }) {
                return Some(found);
            }
        }
    }
    None
}
