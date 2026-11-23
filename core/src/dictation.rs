//! Dictation controller: hotkey / button events → recording → Phonon → paste.
//!
//! One thread owns the current `Recording`. While recording it pushes the mic level to the bar every 50 ms. On stop
//! it resamples to 16 kHz, transcribes with the active model and pastes the text into the focused app.
//! The active model is Phonon (in this process) or, when the user chose one, a Moroccan Darija model on the llama.cpp
//! helper (`models::SpeechBackend`). Darija has no live preview: the bar goes from listening to transcribing.
//!
//! With the live preview on, a second thread re-transcribes the audio so far a few times a second and shows the
//! words above the bar. Old results are dropped as soon as a newer recording starts or this one ends.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::audio::{self, Recording};
use crate::{BarPhase, Event, Sink};
use crate::format::{self, Formatting};
use crate::handsfree::{Action, Input, Machine};
use crate::llm::cleanup::{self, Cleanup};
use crate::llm::asr::AsrModel;
use crate::models::{Models, SpeechBackend};
use crate::prefs::PrefsStore;

/// The model that understands several languages; its output can be locked to one alphabet.
const MULTILINGUAL_MODEL: &str = "phonon-2.5";

/// Shorter presses are treated as accidental taps and dropped.
const MIN_DURATION: Duration = Duration::from_millis(250);
/// Safety cap so a forgotten "Test microphone" can't record forever (shortcut recordings are capped in `handsfree`).
const MAX_DURATION: Duration = Duration::from_secs(300);
const LEVEL_INTERVAL: Duration = Duration::from_millis(50);
/// Live preview: first update after this much audio, then at most this often.
const PREVIEW_AFTER: Duration = Duration::from_millis(500);
const PREVIEW_EVERY: Duration = Duration::from_millis(300);
/// The preview only re-reads the most recent audio, so long dictations stay fast.
const PREVIEW_SECONDS: f32 = 28.0;

/// Current recording's number; preview results for an older one are dropped.
static SESSION: AtomicU64 = AtomicU64::new(0);
/// The most recent transcript, for the bar menu's paste and copy.
static LAST: Mutex<Option<String>> = Mutex::new(None);

pub fn last_text() -> Option<String> {
    LAST.lock().ok().and_then(|l| l.clone())
}

/// A preview transcription is running.
static PREVIEW_BUSY: AtomicBool = AtomicBool::new(false);

struct PreviewJob {
    session: u64,
    audio: audio::Captured,
    /// Seconds of earlier audio not included.
    skipped: f32,
}

pub enum Control {
    /// The shortcut was pressed (when, as seen by the keyboard hook).
    HotkeyDown(Instant),
    /// A shortcut key was released.
    HotkeyUp(Instant),
    /// Esc was pressed while hands-free recording is on (the hook swallowed it).
    Esc,
    /// Started from the app ("Test microphone"); ends with Stop. Shows the text instead of pasting it.
    StartManual,
    /// Stop button on the bar.
    Stop,
    /// Cancel button, or another key pressed during the shortcut.
    Cancel,
}

/// Lets the app take a finished dictation instead of pasting it. Called on the dictation thread with the final text
/// (after rules and formatting); return true when the app has put the text where it belongs, false to paste as usual.
pub type Deliver = Arc<dyn Fn(&str) -> bool + Send + Sync>;

pub struct DictationHandle {
    pub tx: Sender<Control>,
    pub microphone: Arc<Mutex<Option<String>>>,
    /// Text cleanup applied before pasting. Starts as `Formatting::OFF` (no change); the app sets it.
    pub formatting: Arc<Mutex<Formatting>>,
    /// Whether a double-tap of the shortcut locks recording on (hands-free). On by default.
    pub hands_free: Arc<AtomicBool>,
    /// Optional claim on finished text. `None` (the default) means every dictation is pasted into the focused app.
    pub deliver: Arc<Mutex<Option<Deliver>>>,
    /// Smart cleanup (an optional extra pass by a small language model). Off until the app switches it on.
    pub cleanup: Arc<Cleanup>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Finished {
    /// How long the shortcut was held.
    pub duration_ms: u64,
    /// Seconds of audio (at 16 kHz).
    pub audio_seconds: f32,
    /// Loudest sample, 0–1; near zero means the mic captured silence.
    pub peak: f32,
    /// The transcript, or None when no model is loaded.
    pub text: Option<String>,
    pub model: Option<String>,
    pub transcribe_ms: u64,
    /// Whether the text was pasted into another app.
    pub pasted: bool,
    /// Whether the app took the text itself (see `DictationHandle::deliver`) instead of it being pasted.
    pub delivered: bool,
}

struct Current {
    rec: Recording,
    started: Instant,
    manual: bool,
    last_preview: Option<Instant>,
}

fn bar(app: &Ctx, phase: BarPhase, level: f32) {
    (app.sink)(Event::Bar { phase, level });
}

/// What the dictation threads need: where events go and where recordings are kept.
#[derive(Clone)]
struct Ctx {
    sink: Sink,
    data_dir: std::path::PathBuf,
    cleanup: Arc<Cleanup>,
}

fn emit_error(app: &Ctx, message: impl Into<String>) {
    let message = message.into();
    eprintln!("hamsa: {message}");
    (app.sink)(Event::Error(message));
}

pub fn spawn(sink: Sink, data_dir: std::path::PathBuf, models: Arc<Models>, prefs: Arc<PrefsStore>) -> DictationHandle {
    let cleanup = Cleanup::new(&data_dir, sink.clone(), models.llama_host(), models.llm_active_flag());
    let app = Ctx { sink, data_dir, cleanup: cleanup.clone() };
    let (tx, rx) = channel::<Control>();
    let microphone: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let mic = microphone.clone();
    let formatting: Arc<Mutex<Formatting>> = Arc::new(Mutex::new(Formatting::OFF));
    let fmt = formatting.clone();
    let deliver: Arc<Mutex<Option<Deliver>>> = Arc::new(Mutex::new(None));
    let claim = deliver.clone();
    let preview_tx = spawn_preview(app.clone(), models.clone(), prefs.clone());

    let hands_free = Arc::new(AtomicBool::new(true));
    let hf = hands_free.clone();

    std::thread::Builder::new()
        .name("hamsa-dictation".into())
        .spawn(move || {
            let mut current: Option<Current> = None;
            // Tap / double-tap / hold logic for the shortcut; `was_locked` is what the bar and the hook were told last.
            let mut keys = Machine::new();
            let mut was_locked = false;
            loop {
                let msg = if current.is_some() {
                    match rx.recv_timeout(LEVEL_INTERVAL) {
                        Ok(m) => Some(m),
                        Err(RecvTimeoutError::Timeout) => None,
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                } else {
                    match rx.recv() {
                        Ok(m) => Some(m),
                        Err(_) => break,
                    }
                };

                let mut action: Option<Action> = None;
                match msg {
                    // Tick: update the bar's level meter.
                    None => {
                        if let Some(c) = &mut current {
                            bar(&app, BarPhase::Listening, audio::meter(c.rec.rms()));
                            // No point transcribing a tap that may be dropped in a moment.
                            if !keys.pending() {
                                maybe_preview(c, &models, &prefs, &preview_tx);
                            }
                            if c.manual && c.started.elapsed() > MAX_DURATION {
                                action = Some(Action::Finish);
                            }
                        }
                        if action.is_none() {
                            action = keys.step(Instant::now(), Input::Tick);
                        }
                    }
                    Some(Control::StartManual) => {
                        if current.is_none() {
                            current = begin(&app, &models, &mic, true);
                        }
                    }
                    // A "Test microphone" recording is ended by Stop, or by releasing the shortcut as before.
                    Some(Control::HotkeyDown(t)) => {
                        if !current.as_ref().is_some_and(|c| c.manual) {
                            keys.set_enabled(hf.load(Ordering::SeqCst));
                            action = keys.step(t, Input::Down);
                        }
                    }
                    Some(Control::HotkeyUp(t)) => {
                        if current.as_ref().is_some_and(|c| c.manual) {
                            action = Some(Action::Finish);
                        } else {
                            action = keys.step(t, Input::Up);
                        }
                    }
                    Some(Control::Esc) => action = keys.step(Instant::now(), Input::Esc),
                    Some(Control::Stop) => {
                        keys.reset();
                        action = Some(Action::Finish);
                    }
                    Some(Control::Cancel) => {
                        keys.reset();
                        action = Some(Action::Cancel);
                    }
                }

                match action {
                    Some(Action::Start) => {
                        if current.is_none() {
                            current = begin(&app, &models, &mic, false);
                        }
                        if current.is_none() {
                            keys.reset();
                        }
                    }
                    Some(Action::Restart) => {
                        if let Some(c) = current.take() {
                            SESSION.fetch_add(1, Ordering::SeqCst);
                            let _ = c.rec.finish();
                        }
                        current = begin(&app, &models, &mic, false);
                        if current.is_none() {
                            keys.reset();
                        }
                    }
                    Some(Action::Lock) => {
                        if current.is_none() {
                            keys.reset();
                        }
                    }
                    Some(Action::Finish) => {
                        // Tell the bar first: its lock goes away as the pill turns to "transcribing".
                        sync_lock(&app, false, &mut was_locked);
                        if let Some(c) = current.take() {
                            SESSION.fetch_add(1, Ordering::SeqCst);
                            finish(&app, &models, &prefs, &fmt, &claim, c);
                        }
                    }
                    Some(Action::Cancel) => {
                        if let Some(c) = current.take() {
                            SESSION.fetch_add(1, Ordering::SeqCst);
                            let _ = c.rec.finish();
                        }
                        sync_lock(&app, false, &mut was_locked);
                        bar(&app, BarPhase::Idle, 0.0);
                    }
                    // A lone tap: dropped without a word, no error and no history entry.
                    Some(Action::Discard) => {
                        if let Some(c) = current.take() {
                            SESSION.fetch_add(1, Ordering::SeqCst);
                            let _ = c.rec.finish();
                        }
                        bar(&app, BarPhase::Idle, 0.0);
                    }
                    None => {}
                }
                sync_lock(&app, keys.locked() && current.is_some(), &mut was_locked);
            }
        })
        .expect("spawn dictation thread");

    DictationHandle { tx, microphone, formatting, deliver, hands_free, cleanup }
}

/// Start recording; on failure show the error and return None.
fn begin(app: &Ctx, models: &Models, mic: &Mutex<Option<String>>, manual: bool) -> Option<Current> {
    let device = mic.lock().ok().and_then(|m| m.clone());
    match audio::start(device) {
        Ok(rec) => {
            SESSION.fetch_add(1, Ordering::SeqCst);
            // Load the Darija model, or else the cleanup model, now so it is ready when the words are.
            if !models.warm_up_speech() {
                app.cleanup.warm_up();
            }
            bar(app, BarPhase::Listening, 0.0);
            Some(Current { rec, started: Instant::now(), manual, last_preview: None })
        }
        Err(e) => {
            emit_error(app, e);
            bar(app, BarPhase::Idle, 0.0);
            None
        }
    }
}

/// Tell the bar (lock icon) and the keyboard hook (Esc is swallowed only while locked) when hands-free changes.
fn sync_lock(app: &Ctx, locked: bool, was: &mut bool) {
    if locked != *was {
        *was = locked;
        #[cfg(windows)]
        crate::hotkey::set_locked(locked);
        (app.sink)(Event::Locked(locked));
    }
}

fn finish(app: &Ctx, models: &Models, prefs: &PrefsStore, fmt: &Mutex<Formatting>, claim: &Mutex<Option<Deliver>>, c: Current) {
    let held = c.started.elapsed();
    let captured = match c.rec.finish() {
        Ok(cap) => cap,
        Err(e) => {
            emit_error(app, e);
            bar(app, BarPhase::Idle, 0.0);
            return;
        }
    };
    if held < MIN_DURATION {
        bar(app, BarPhase::Idle, 0.0);
        return;
    }

    bar(app, BarPhase::Transcribing, 0.0);
    let result = (|| -> Result<Finished, String> {
        let pcm = audio::to_16k(&captured.samples, captured.sample_rate)?;
        let peak = pcm.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        // Keep the latest recording for troubleshooting.
        {
            let dir = &app.data_dir;
            let _ = audio::save_wav(&dir.join("recordings").join("last.wav"), &pcm);
        }

        let (mut text, mut model, mut transcribe_ms, mut pasted) = (None, None, 0, false);
        let mut delivered = false;
        if let Some(backend) = models.backend() {
            let id = backend.id();
            let darija = matches!(backend, SpeechBackend::Llama(_));
            let t = Instant::now();
            let (mut transcript, snippet) = match &backend {
                SpeechBackend::Phonon { id, engine } => transcribe_detailed(id, engine, prefs, &pcm),
                SpeechBackend::Llama(m) => transcribe_darija(models, *m, prefs, &pcm),
            };
            // A snippet is pasted exactly as saved; everything else gets the Style page's cleanup.
            if !snippet {
                let f = fmt.lock().map(|f| *f).unwrap_or(Formatting::OFF);
                let listed;
                (transcript, listed) = format::apply_detailed(&transcript, &f);
                if listed && app.cleanup.active() {
                    // The rewrite would flatten the lines, so a list is pasted as the rules made it.
                    eprintln!("hamsa: smart cleanup skipped, the text was formatted as a list");
                }
                // Optional smart cleanup (English only, never for Darija); the bar stays on "transcribing" meanwhile.
                if !listed && !darija && !transcript.is_empty() && app.cleanup.active() && cleanup::english_gate(&prefs.get().language, &id, &transcript) {
                    transcript = app.cleanup.clean(&transcript);
                }
            }
            transcribe_ms = t.elapsed().as_millis() as u64;
            if !c.manual && !transcript.is_empty() {
                // The app may take the text itself (e.g. Hamsa's own Notes page); otherwise it is pasted.
                let taker = claim.lock().ok().and_then(|d| d.clone());
                delivered = taker.is_some_and(|take| take(&transcript));
                if !delivered {
                    wait_for_shortcut_release();
                    match crate::paste::paste_text(&transcript) {
                        Ok(()) => pasted = true,
                        Err(e) => emit_error(app, format!("Couldn't paste the text: {e}")),
                    }
                }
            }
            if !transcript.is_empty() {
                if let Ok(mut l) = LAST.lock() {
                    *l = Some(transcript.clone());
                }
            }
            text = Some(transcript);
            model = Some(id);
        }
        Ok(Finished {
            duration_ms: held.as_millis() as u64,
            audio_seconds: pcm.len() as f32 / audio::TARGET_RATE as f32,
            peak,
            text,
            model,
            transcribe_ms,
            pasted,
            delivered,
        })
    })();

    match result {
        Ok(done) => {
            (app.sink)(Event::Finished(done));
        }
        Err(e) => emit_error(app, format!("Couldn't process the recording: {e}")),
    }
    bar(app, BarPhase::Idle, 0.0);
}

/// Ctrl+V must not be pressed while shortcut keys are still held (Win would turn it into Ctrl+Win+V).
fn wait_for_shortcut_release() {
    #[cfg(windows)]
    {
        let start = Instant::now();
        while crate::hotkey::shortcut_held() && start.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// Transcribe with the user's settings: language alphabet, dictionary boost, then dictionary and snippets.
fn transcribe(model_id: &str, engine: &phonon::Engine, prefs: &PrefsStore, pcm: &[f32]) -> String {
    transcribe_detailed(model_id, engine, prefs, pcm).0
}

/// As `transcribe`; the flag says the whole utterance was one snippet cue.
fn transcribe_detailed(model_id: &str, engine: &phonon::Engine, prefs: &PrefsStore, pcm: &[f32]) -> (String, bool) {
    let p = prefs.get();
    // The multilingual model can be held to the chosen language's alphabet.
    let script = if model_id == MULTILINGUAL_MODEL { phonon::script_for_language(&p.language) } else { None };
    let boost = engine.boost(&p.vocabulary.terms());
    let raw = engine.transcribe_opts(pcm, &phonon::Options { script, boost: Some(&boost) }).text;
    p.vocabulary.apply_detailed(&raw)
}

/// Transcribe with a Darija model. Whatever goes wrong (helper not ready, timeout, a model that never stops) gives an
/// empty text, which the app shows as "No words recognised"; nothing is ever pasted from a failed run.
/// The dictionary's replacements and snippets apply as for Phonon.
fn transcribe_darija(models: &Models, model: AsrModel, prefs: &PrefsStore, pcm: &[f32]) -> (String, bool) {
    match models.asr().transcribe(model, pcm) {
        Ok(done) => {
            if let Some(backend) = done.backend {
                eprintln!("hamsa: {} ran on the {} in {} ms", model.name(), backend.label(), done.ms);
            }
            prefs.get().vocabulary.apply_detailed(&done.text)
        }
        Err(e) => {
            eprintln!("hamsa: {}: {e}", model.name());
            (String::new(), false)
        }
    }
}

/// Queue a preview transcription if it's time and none is running. Darija models have no live preview.
fn maybe_preview(c: &mut Current, models: &Models, prefs: &PrefsStore, tx: &Sender<PreviewJob>) {
    if models.speech_override().is_some() {
        return;
    }
    let due = match c.last_preview {
        None => c.started.elapsed() >= PREVIEW_AFTER,
        Some(t) => t.elapsed() >= PREVIEW_EVERY,
    };
    if !due || PREVIEW_BUSY.load(Ordering::SeqCst) || models.engine().is_none() || !prefs.get().bar.preview {
        return;
    }
    if let Some((audio, skipped)) = c.rec.snapshot(PREVIEW_SECONDS) {
        PREVIEW_BUSY.store(true, Ordering::SeqCst);
        c.last_preview = Some(Instant::now());
        let _ = tx.send(PreviewJob { session: SESSION.load(Ordering::SeqCst), audio, skipped });
    }
}

fn spawn_preview(app: Ctx, models: Arc<Models>, prefs: Arc<PrefsStore>) -> Sender<PreviewJob> {
    let (tx, rx): (Sender<PreviewJob>, Receiver<PreviewJob>) = channel();
    std::thread::Builder::new()
        .name("hamsa-preview".into())
        .spawn(move || {
            let mut last: (u64, Vec<String>) = (0, Vec::new());
            for job in rx {
                let current = || SESSION.load(Ordering::SeqCst) == job.session;
                if current() {
                    if let (Some((id, engine)), Ok(pcm)) =
                        (models.engine(), audio::to_16k(&job.audio.samples, job.audio.sample_rate))
                    {
                        let text = transcribe(&id, &engine, &prefs, &pcm);
                        let words: Vec<String> = text.split_whitespace().map(String::from).collect();
                        if last.0 != job.session {
                            last = (job.session, Vec::new());
                        }
                        let (stable, tentative) = split_stable(&last.1, &words, job.skipped > 0.0);
                        if current() {
                            (app.sink)(Event::Preview { stable: stable.clone(), tentative: tentative.clone() });
                        }
                        last.1 = words;
                    }
                }
                // Pace the next update from when this one finished.
                PREVIEW_BUSY.store(false, Ordering::SeqCst);
            }
        })
        .expect("spawn preview thread");
    tx
}

/// Words that match the previous preview are settled; the rest may still change.
/// When the start of the audio has scrolled out (`sliding`), only the last two words count as unsettled.
fn split_stable(prev: &[String], now: &[String], sliding: bool) -> (String, String) {
    let settled = if sliding {
        now.len().saturating_sub(2)
    } else {
        prev.iter().zip(now).take_while(|(a, b)| a == b).count()
    };
    (now[..settled].join(" "), now[settled..].join(" "))
}

#[cfg(test)]
mod tests {
    use super::split_stable;

    fn w(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn settled_words_are_the_common_start() {
        assert_eq!(split_stable(&w("can we move"), &w("can we move the review"), false), ("can we move".into(), "the review".into()));
        assert_eq!(split_stable(&w("can we moo"), &w("can we move it"), false), ("can we".into(), "move it".into()));
        assert_eq!(split_stable(&[], &w("hello there"), false), ("".into(), "hello there".into()));
        assert_eq!(split_stable(&[], &w("a b c d"), true), ("a b".into(), "c d".into()));
    }
}
