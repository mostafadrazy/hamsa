//! Moroccan Darija speech recognition: the fihrIA-ASR models (a Qwen3-ASR fine-tune) run by the llama.cpp helper.
//!
//! Two models, "Darija · fast" (0.6B) and "Darija · accurate" (1.7B). Each is two files from Hugging Face, pinned to a
//! commit and a SHA-256: the language model (Q4_K_M) and its audio projector (`mmproj`). They live in
//! `<data>\llm\<id>\`, never in the shared `models` folder.
//!
//! The recording (16 kHz mono) is sent to `llama-server` as a WAV, base64 in an OpenAI-style `input_audio` part, after
//! an empty system message and with nothing else in the user turn. The answer looks like
//! `language Darija<asr_text>…`; the prefix is removed. `parse_output` does that.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{Backend, HostState, LlamaHost};

/// A speech model run by the llama.cpp helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AsrModel {
    /// fihrIA-ASR 0.6B.
    Fast,
    /// fihrIA-ASR 1.7B.
    Accurate,
}

/// One pinned file.
pub struct Part {
    pub file: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

impl AsrModel {
    pub const ALL: [AsrModel; 2] = [AsrModel::Fast, AsrModel::Accurate];

    /// Also the name of the model's folder under `llm`.
    pub fn id(self) -> &'static str {
        match self {
            AsrModel::Fast => "fihria-asr-0.6b",
            AsrModel::Accurate => "fihria-asr-1.7b",
        }
    }

    pub fn from_id(id: &str) -> Option<AsrModel> {
        AsrModel::ALL.into_iter().find(|m| m.id() == id)
    }

    pub fn name(self) -> &'static str {
        match self {
            AsrModel::Fast => "Darija · fast",
            AsrModel::Accurate => "Darija · accurate",
        }
    }

    fn repo(self) -> &'static str {
        match self {
            AsrModel::Fast => "fihriaLab/fihrIA-ASR-0.6B-GGUF",
            AsrModel::Accurate => "fihriaLab/fihrIA-ASR-1.7B-GGUF",
        }
    }

    /// The commit of the Hugging Face repo the files are pinned to.
    pub fn commit(self) -> &'static str {
        match self {
            AsrModel::Fast => "7746d5495e0a5e0636b750d96f9182c30d6dfc4d",
            AsrModel::Accurate => "384b430b9529c820488e0c05a6835a62b8c499db",
        }
    }

    /// The language model, Q4_K_M (SHA-256 = the file's Hugging Face LFS id).
    pub fn llm(self) -> Part {
        match self {
            AsrModel::Fast => Part {
                file: "fihria-asr-0.6b-Q4_K_M.gguf",
                bytes: 342_106_720,
                sha256: "1d73087f5eb6089dd232dcd4e5678871ea1344529b462e76b9d206c1f38b2e8d",
            },
            AsrModel::Accurate => Part {
                file: "fihria-asr-1.7b-Q4_K_M.gguf",
                bytes: 999_944_608,
                sha256: "f7cd488acee8dc3379bf51dedae0e698a7388b44d94e3b327ed22fd839b3850e",
            },
        }
    }

    /// The audio projector.
    pub fn mmproj(self) -> Part {
        match self {
            AsrModel::Fast => Part {
                file: "mmproj-fihria-asr-0.6b-BF16.gguf",
                bytes: 378_575_520,
                sha256: "2391de296c0ae41d578a3eb9f946e81c245ea52043d9edbb0cb1075e5095ef35",
            },
            AsrModel::Accurate => Part {
                file: "mmproj-fihria-asr-1.7b-BF16.gguf",
                bytes: 641_773_952,
                sha256: "63be5803069b43c290e03b78833e81c2343c75cde5969c6370b826a3341fd65a",
            },
        }
    }

    /// Both files together.
    pub fn total_bytes(self) -> u64 {
        self.llm().bytes + self.mmproj().bytes
    }

    pub fn url(self, part: &Part) -> String {
        format!("https://huggingface.co/{}/resolve/{}/{}", self.repo(), self.commit(), part.file)
    }

    /// Up to this many seconds of recording fit in the server's context (see `max_tokens`).
    pub fn max_seconds(self) -> f32 {
        (CONTEXT as f32 - 160.0) / AUDIO_TOKENS_PER_SECOND
    }
}

/// Must match the `-c` the helper uses for audio models.
const CONTEXT: i64 = 8192;
/// Qwen3-ASR's audio encoder makes 12.5 tokens per second of sound.
const AUDIO_TOKENS_PER_SECOND: f32 = 12.5;
/// How long a dictation waits for a model that is still loading.
const START_WAIT: Duration = Duration::from_secs(60);
/// Recordings shorter than this, or quieter than `MIN_PEAK`, are not sent (the model would make words up).
const MIN_SECONDS: f32 = 0.25;
const MIN_PEAK: f32 = 0.002;

/// A finished transcription.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// The server's raw answer (before the `language …` prefix is removed), for the developer hook.
    pub raw: String,
    /// Time spent on the request (not on loading the model).
    pub ms: u64,
    pub backend: Option<Backend>,
}

pub struct Asr {
    data_dir: PathBuf,
    host: Arc<LlamaHost>,
}

impl Asr {
    pub fn new(data_dir: &Path, host: Arc<LlamaHost>) -> Arc<Self> {
        Arc::new(Self { data_dir: data_dir.to_path_buf(), host })
    }

    pub fn host(&self) -> &Arc<LlamaHost> {
        &self.host
    }

    pub fn dir(&self, m: AsrModel) -> PathBuf {
        self.data_dir.join("llm").join(m.id())
    }

    pub fn llm_path(&self, m: AsrModel) -> PathBuf {
        self.dir(m).join(m.llm().file)
    }

    pub fn mmproj_path(&self, m: AsrModel) -> PathBuf {
        self.dir(m).join(m.mmproj().file)
    }

    /// Both files are there, complete (a half-downloaded file is only ever a `.part`).
    pub fn installed(&self, m: AsrModel) -> bool {
        let ok = |path: PathBuf, part: Part| fs::metadata(path).is_ok_and(|f| f.len() == part.bytes);
        ok(self.llm_path(m), m.llm()) && ok(self.mmproj_path(m), m.mmproj())
    }

    /// What the helper is doing with this model; `None` when it is serving another one (or nothing).
    pub fn host_state(&self, m: AsrModel) -> Option<HostState> {
        (self.host.model().as_deref() == Some(self.llm_path(m).as_path())).then(|| self.host.state())
    }

    /// Start loading `m` in the background, so it is ready when the recording is.
    pub fn warm_up(&self, m: AsrModel) {
        if self.installed(m) {
            self.host.start_with(&self.llm_path(m), Some(&self.mmproj_path(m)), Some(warmup_body()));
        }
    }

    /// Stop the helper if it is serving one of these models.
    pub fn stop_if_serving(&self) {
        if self.host.model().is_some_and(|p| AsrModel::ALL.iter().any(|m| p == self.llm_path(*m))) {
            self.host.stop();
        }
    }

    /// Download llama.cpp (if missing) and both files of `m`, reporting overall progress (0–1, about every percent).
    pub fn fetch(&self, m: AsrModel, progress: &mut dyn FnMut(f32)) -> Result<(), String> {
        let llama_needed = !super::llama_installed(&self.data_dir);
        let llama_bytes = if llama_needed { super::LLAMA_BYTES } else { 0 };
        let total = (llama_bytes + m.total_bytes()) as f64;
        let mut last = -1i64;
        let mut report = |done: u64, progress: &mut dyn FnMut(f32)| {
            let pct = (done as f64 * 100.0 / total) as i64;
            if pct != last {
                last = pct;
                progress((done as f64 / total).min(1.0) as f32);
            }
        };
        if llama_needed {
            super::install_llama(&self.data_dir, &mut |n| report(n, progress))?;
        }
        let mut done = llama_bytes;
        for (path, part) in [(self.llm_path(m), m.llm()), (self.mmproj_path(m), m.mmproj())] {
            if !fs::metadata(&path).is_ok_and(|f| f.len() == part.bytes) {
                let base = done;
                super::download_verified(&m.url(&part), &path, part.sha256, part.bytes, &mut |n| report(base + n, progress))?;
            }
            done += part.bytes;
        }
        Ok(())
    }

    /// Delete the model's files (the llama.cpp helper stays; other features share it).
    pub fn remove(&self, m: AsrModel) {
        if self.host.model().as_deref() == Some(self.llm_path(m).as_path()) {
            self.host.stop();
        }
        let _ = fs::remove_dir_all(self.dir(m));
    }

    /// Transcribe 16 kHz mono audio. Silence and taps give an empty text without asking the model.
    pub fn transcribe(&self, m: AsrModel, pcm: &[f32]) -> Result<Transcript, String> {
        let seconds = pcm.len() as f32 / 16_000.0;
        let peak = pcm.iter().fold(0.0f32, |p, s| p.max(s.abs()));
        if seconds < MIN_SECONDS || peak < MIN_PEAK {
            return Ok(Transcript { text: String::new(), raw: String::new(), ms: 0, backend: None });
        }
        self.transcribe_unguarded(m, pcm)
    }

    /// As `transcribe`, but always asks the model (the developer hook uses it to see what silence produces).
    pub fn transcribe_unguarded(&self, m: AsrModel, pcm: &[f32]) -> Result<Transcript, String> {
        let seconds = pcm.len() as f32 / 16_000.0;
        let max_tokens = max_tokens(seconds)?;
        let host = &self.host;
        let (port, backend) =
            host.wait_ready_with(&self.llm_path(m), Some(&self.mmproj_path(m)), Some(warmup_body()), START_WAIT)?;
        let started = Instant::now();
        let body = request_body(&base64(&wav_bytes(pcm)), max_tokens);
        let reply = host.post_json(port, "/v1/chat/completions", &body, request_timeout(seconds, backend))?;
        let (raw, finish) = parse_reply(&reply)?;
        if finish == "length" {
            // It never found the end of the speech (repeating itself): that is not a transcript.
            return Err("the model did not stop talking".into());
        }
        Ok(Transcript { text: parse_output(&raw), raw, ms: started.elapsed().as_millis() as u64, backend: Some(backend) })
    }
}

/// How long the request may take: more audio takes longer, and the CPU is slower than a graphics card.
pub fn request_timeout(audio_seconds: f32, backend: Backend) -> Duration {
    let (floor, factor) = if backend == Backend::Gpu { (10.0, 1.5) } else { (20.0, 4.0) };
    Duration::from_secs_f32((audio_seconds * factor).max(floor))
}

/// The most words the model may answer with: generous (about 8 tokens per second of speech), but never more than
/// what is left of the context after the audio. Too much audio for the context is an error.
pub fn max_tokens(audio_seconds: f32) -> Result<u32, String> {
    let audio_tokens = (audio_seconds * AUDIO_TOKENS_PER_SECOND).ceil() as i64 + 64;
    let room = CONTEXT - audio_tokens - 32;
    if room < 64 {
        return Err(format!("the recording is too long for this model ({audio_seconds:.0} s)"));
    }
    Ok(((audio_seconds * 8.0) as i64 + 64).min(room).min(3000) as u32)
}

/// The chat-completions request: an empty system message and a user turn holding only the audio.
pub fn request_body(audio_base64: &str, max_tokens: u32) -> String {
    serde_json::json!({
        "model": "fihria-asr",
        "messages": [
            { "role": "system", "content": "" },
            { "role": "user", "content": [
                { "type": "input_audio", "input_audio": { "data": audio_base64, "format": "wav" } }
            ] },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": false,
    })
    .to_string()
}

/// Sent once after the model loads: one second of silence, so the first dictation doesn't pay for building the graphs.
fn warmup_body() -> String {
    request_body(&base64(&wav_bytes(&vec![0.0f32; 16_000])), 8)
}

/// The answer text and why it ended ("stop" or "length").
fn parse_reply(body: &str) -> Result<(String, String), String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("unreadable response: {e}"))?;
    let choice = &v["choices"][0];
    let text = choice["message"]["content"].as_str().ok_or("response has no text")?;
    Ok((text.to_string(), choice["finish_reason"].as_str().unwrap_or("stop").to_string()))
}

const LANGUAGE_NAMES: [&str; 6] = ["Darija", "Arabic", "French", "English", "Amazigh", "None"];

/// The transcript out of the model's answer: removes the leading `language Darija` (and the `<asr_text>` marker
/// that may follow it). The language name is followed directly by the Arabic words, so it is matched against the
/// known names first and as a run of Latin letters otherwise; a word of the transcript is never taken for it.
pub fn parse_output(raw: &str) -> String {
    let t = raw.trim();
    let mut rest = t;
    let mut language_none = false;
    if let Some(after) = t.strip_prefix("language") {
        if after.starts_with(char::is_whitespace) {
            let after = after.trim_start();
            let len = LANGUAGE_NAMES
                .iter()
                .find(|n| after.starts_with(*n))
                .map(|n| n.len())
                .unwrap_or_else(|| after.chars().take_while(|c| c.is_ascii_alphabetic()).count());
            if len > 0 {
                language_none = after[..len].eq_ignore_ascii_case("none");
                rest = after[len..].trim_start();
            }
        }
    }
    let text = rest.replace("<asr_text>", "");
    let text = text.trim();
    // "language None" means no speech was found.
    if language_none && text.is_empty() {
        return String::new();
    }
    text.to_string()
}

/// 16 kHz mono PCM as a 16-bit WAV file in memory.
pub fn wav_bytes(pcm: &[f32]) -> Vec<u8> {
    let data_len = (pcm.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + pcm.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&32_000u32.to_le_bytes()); // bytes per second
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes());
    }
    out
}

pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { ALPHABET[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Developer hook (`HAMSA_BETA_ASR_TEST` = a WAV file, `HAMSA_BETA_ASR_MODEL` = 0.6b or 1.7b): transcribe the file with
/// that model, print the text, the timings and where it ran, write `asr-test.log`, stop the helper. Downloads the
/// model first if it is missing.
pub fn dev_test(wav: &Path, m: AsrModel) {
    let data_dir = crate::data_dir();
    let asr = Asr::new(&data_dir, LlamaHost::new(&data_dir));
    let log_path = data_dir.join("asr-test.log");
    let mut log = String::new();
    let mut say = |line: String| {
        println!("{line}");
        log.push_str(&line);
        log.push('\n');
    };
    say(format!("model: {} ({}), {}", m.name(), m.id(), m.llm().file));
    let pcm = match read_wav(wav) {
        Ok(p) => p,
        Err(e) => {
            say(format!("can't read the WAV: {e}"));
            let _ = fs::write(&log_path, log);
            return;
        }
    };
    let peak = pcm.iter().fold(0.0f32, |p, s| p.max(s.abs()));
    say(format!("audio: {} ({:.2} s, peak {:.3})", wav.display(), pcm.len() as f32 / 16_000.0, peak));
    if !asr.installed(m) || !super::llama_installed(&data_dir) {
        say("downloading llama.cpp and the model...".into());
        let t = Instant::now();
        let mut last = -1i64;
        let result = asr.fetch(m, &mut |p| {
            let pct = (p * 100.0) as i64;
            if pct / 10 != last / 10 {
                println!("  {pct}%");
            }
            last = pct;
        });
        match result {
            Ok(()) => say(format!("downloaded in {:.1} s", t.elapsed().as_secs_f32())),
            Err(e) => {
                say(format!("download failed: {e}"));
                let _ = fs::write(&log_path, log);
                return;
            }
        }
    }
    let t = Instant::now();
    let ready = asr.host.wait_ready_with(&asr.llm_path(m), Some(&asr.mmproj_path(m)), Some(warmup_body()), Duration::from_secs(120));
    say(format!("load (with warm-up): {} ms, {}", t.elapsed().as_millis(), match &ready {
        Ok((port, b)) => format!("{} on port {port}", b.label()),
        Err(e) => format!("failed: {e}"),
    }));
    if ready.is_ok() {
        for round in 1..=2 {
            match asr.transcribe_unguarded(m, &pcm) {
                Ok(done) => {
                    say(format!("run {round} raw:  {}", done.raw));
                    say(format!("run {round} text: {}", done.text));
                    say(format!(
                        "run {round}: {} ms on the {}",
                        done.ms,
                        done.backend.map_or("?", |b| b.label())
                    ));
                }
                Err(e) => say(format!("run {round} failed: {e}")),
            }
        }
        // What dictation itself would do (silence and taps are not sent to the model).
        match asr.transcribe(m, &pcm) {
            Ok(done) if done.backend.is_none() => say("dictation: too short or silent, not sent (empty text)".into()),
            Ok(done) => say(format!("dictation: {}", done.text)),
            Err(e) => say(format!("dictation: failed ({e}), nothing would be pasted")),
        }
    }
    asr.host.stop();
    let _ = fs::write(&log_path, log);
}

/// Any WAV file as 16 kHz mono samples.
fn read_wav(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>().map_err(|e| e.to_string())?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>().map_err(|e| e.to_string())?
        }
    };
    let mono: Vec<f32> = samples.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect();
    crate::audio::to_16k(&mono, spec.sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_language_prefix_is_removed() {
        assert_eq!(parse_output("language Darija<asr_text>التلفون ديالي طاح وتهرس ليه l'écran"), "التلفون ديالي طاح وتهرس ليه l'écran");
        // The server's answer has no marker, and the Arabic follows the name directly or after a space.
        assert_eq!(parse_output("language Darijaالتلفون ديالي"), "التلفون ديالي");
        assert_eq!(parse_output("language Darija التلفون ديالي"), "التلفون ديالي");
        assert_eq!(parse_output("  language Darija<asr_text> سلام  \n"), "سلام");
        // A French word right after the name is not swallowed into it.
        assert_eq!(parse_output("language Darijabonjour tout le monde"), "bonjour tout le monde");
        assert_eq!(parse_output("language Arabic<asr_text>مرحبا"), "مرحبا");
        // Another language name still goes.
        assert_eq!(parse_output("language Hindi<asr_text>नमस्ते"), "नमस्ते");
    }

    #[test]
    fn answers_without_a_prefix_are_kept() {
        assert_eq!(parse_output("سلام labas"), "سلام labas");
        assert_eq!(parse_output("<asr_text>سلام"), "سلام");
        assert_eq!(parse_output("languages are fun"), "languages are fun", "not the prefix: no space after `language`");
        assert_eq!(parse_output("language"), "language");
        assert_eq!(parse_output(""), "");
    }

    #[test]
    fn no_speech_is_an_empty_text() {
        assert_eq!(parse_output("language None<asr_text>"), "");
        assert_eq!(parse_output("language None"), "");
        assert_eq!(parse_output("language Darija<asr_text>"), "");
    }

    #[test]
    fn the_request_is_an_empty_system_message_and_audio_only() {
        let v: serde_json::Value = serde_json::from_str(&request_body("QUJD", 100)).unwrap();
        assert_eq!(v["messages"][0]["role"], "system");
        assert_eq!(v["messages"][0]["content"], "");
        assert_eq!(v["messages"][1]["role"], "user");
        let parts = v["messages"][1]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 1, "audio only");
        assert_eq!(parts[0]["type"], "input_audio");
        assert_eq!(parts[0]["input_audio"]["data"], "QUJD");
        assert_eq!(parts[0]["input_audio"]["format"], "wav");
        assert_eq!(v["temperature"], 0);
        assert_eq!(v["max_tokens"], 100);
        assert_eq!(v["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn timeouts_follow_the_audio_and_the_device() {
        assert_eq!(request_timeout(2.0, Backend::Gpu), Duration::from_secs(10));
        assert_eq!(request_timeout(20.0, Backend::Gpu), Duration::from_secs(30));
        assert_eq!(request_timeout(2.0, Backend::Cpu), Duration::from_secs(20));
        assert_eq!(request_timeout(10.0, Backend::Cpu), Duration::from_secs(40));
    }

    #[test]
    fn the_answer_length_is_capped_by_the_context() {
        assert_eq!(max_tokens(3.0).unwrap(), 88);
        assert!(max_tokens(60.0).unwrap() <= 3000);
        assert!(max_tokens(700.0).is_err());
        let longest = AsrModel::Fast.max_seconds();
        assert!(max_tokens(longest - 1.0).is_ok());
        assert!(max_tokens(longest + 20.0).is_err());
    }

    #[test]
    fn replies_are_read_with_their_end_reason() {
        let ok = r#"{"choices":[{"message":{"content":"language Darija<asr_text>سلام"},"finish_reason":"stop"}]}"#;
        assert_eq!(parse_reply(ok).unwrap(), ("language Darija<asr_text>سلام".to_string(), "stop".to_string()));
        let cut = r#"{"choices":[{"message":{"content":"x"},"finish_reason":"length"}]}"#;
        assert_eq!(parse_reply(cut).unwrap().1, "length");
        assert!(parse_reply("nope").is_err());
        assert!(parse_reply(r#"{"error":{"message":"x"}}"#).is_err());
    }

    #[test]
    fn wav_and_base64_are_well_formed() {
        let wav = wav_bytes(&[0.0, 1.0, -1.0, 0.5]);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(wav.len(), 44 + 8);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 8);
        assert_eq!(i16::from_le_bytes(wav[46..48].try_into().unwrap()), i16::MAX);
        assert_eq!(i16::from_le_bytes(wav[48..50].try_into().unwrap()), -i16::MAX);
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn models_are_pinned() {
        for m in AsrModel::ALL {
            assert_eq!(AsrModel::from_id(m.id()), Some(m));
            assert_eq!(m.commit().len(), 40);
            for part in [m.llm(), m.mmproj()] {
                assert_eq!(part.sha256.len(), 64);
                assert!(m.url(&part).contains(m.commit()) && m.url(&part).ends_with(part.file));
            }
            assert_eq!(m.total_bytes(), m.llm().bytes + m.mmproj().bytes);
        }
        assert_eq!(AsrModel::from_id("phonon-2.5"), None);
        // About 720 MB and 1.64 GB, as the cards say.
        assert_eq!(AsrModel::Fast.total_bytes() / 1_000_000, 720);
        assert_eq!(AsrModel::Accurate.total_bytes() / 10_000_000, 164);
    }

    #[test]
    fn silence_and_taps_are_not_sent_to_the_model() {
        let data = std::env::temp_dir().join(format!("hamsa-asr-test-{}", std::process::id()));
        let asr = Asr::new(&data, LlamaHost::new(&data));
        let quiet = asr.transcribe(AsrModel::Fast, &vec![0.0; 32_000]).unwrap();
        assert!(quiet.text.is_empty() && quiet.backend.is_none());
        let tap = asr.transcribe(AsrModel::Fast, &vec![0.5; 1_000]).unwrap();
        assert!(tap.text.is_empty() && tap.backend.is_none());
        // Real audio with no model installed is an error, never a made-up text.
        assert!(asr.transcribe_unguarded(AsrModel::Fast, &vec![0.5; 32_000]).is_err());
    }
}
