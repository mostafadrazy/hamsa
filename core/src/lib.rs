//! Hamsa's app logic without any UI: microphone, hold-to-talk shortcut, speech models, dictation, dictionary and
//! snippets, preferences, pasting. A UI (Hamsa Beta's Slint windows) receives `Event`s through a `Sink` and calls
//! the functions here.
//!
//! Data lives in `%LOCALAPPDATA%\app.hamsa.desktop`, shared with older Hamsa builds (same preferences and models).

pub mod audio;
pub mod format;
pub mod handsfree;
#[cfg(windows)]
pub mod hotkey;
pub mod layout;
mod lists;
pub mod llm;
pub mod models;
pub mod paste;
pub mod prefs;
pub mod vocabulary;
pub mod dictation;

use std::path::PathBuf;
use std::sync::Arc;

pub use dictation::{Control, Deliver, DictationHandle, Finished};
pub use format::{Bullet, Formatting};
pub use layout::{Layout, Placement};
pub use llm::cleanup::{Cleanup, CleanupStatus, LlmState, Size as CleanupSize};
pub use llm::asr::AsrModel;
pub use models::{ModelStatus, Models, SpeechBackend};
pub use prefs::{Prefs, PrefsStore};
pub use vocabulary::{Rule, Vocabulary};

/// What the dictation bar is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarPhase {
    Idle,
    Listening,
    Transcribing,
}

/// Things the UI should show. Sent from background threads.
#[derive(Clone, Debug)]
pub enum Event {
    /// Bar state and mic level (0â€“1, only while listening).
    Bar { phase: BarPhase, level: f32 },
    /// Live preview words: `stable` stayed the same between updates, `tentative` may still change.
    Preview { stable: String, tentative: String },
    /// Model list changed (download progress, loading, active model, device).
    Models(Vec<ModelStatus>),
    /// Smart cleanup changed: switch, size, download progress, helper state.
    Cleanup(CleanupStatus),
    /// A dictation finished.
    Finished(Finished),
    /// Something went wrong (microphone, paste, ...).
    Error(String),
    /// Hands-free (double-tap) recording turned on or off; the bar shows a lock while it is on.
    Locked(bool),
    /// Shortcut capture result: key ids (empty = cancelled with Esc).
    ShortcutCaptured(Vec<u16>),
}

/// Where events go. Must be cheap and thread-safe; the UI forwards them to its event loop.
pub type Sink = Arc<dyn Fn(Event) + Send + Sync>;

/// `%LOCALAPPDATA%\app.hamsa.desktop` (created if missing).
pub fn data_dir() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    let dir = base.join("app.hamsa.desktop");
    let _ = std::fs::create_dir_all(&dir);
    dir
}
