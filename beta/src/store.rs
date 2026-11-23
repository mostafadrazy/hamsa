//! Hamsa Beta's own files in the shared data folder:
//! - `history.json`: past dictations (the Tauri build keeps its history inside its web view instead).
//! - `beta.json`: appearance, microphone choice, hands-free and smart cleanup.
//! - `notes.json`: the Notes page.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dictation {
    pub text: String,
    /// Unix time in milliseconds.
    pub at: i64,
    pub words: u32,
    pub duration_ms: u64,
}

pub struct History {
    path: PathBuf,
    pub items: Vec<Dictation>,
}

const MAX_HISTORY: usize = 1000;

impl History {
    pub fn load(dir: &Path) -> Self {
        let path = dir.join("history.json");
        let items = fs::read(&path).ok().and_then(|raw| serde_json::from_slice(&raw).ok()).unwrap_or_default();
        Self { path, items }
    }

    pub fn add(&mut self, d: Dictation) {
        self.items.insert(0, d);
        self.items.truncate(MAX_HISTORY);
        save(&self.path, &self.items);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BetaSettings {
    /// "system" | "light" | "dark"
    pub theme: String,
    /// Microphone name; empty = Windows default.
    pub microphone: String,
    /// Double-tap the shortcut to keep listening without holding it. On unless switched off.
    pub hands_free: bool,
    /// Smart cleanup (a small language model tidies the text). Off unless switched on.
    pub smart_cleanup: bool,
    /// Which model file smart cleanup uses: "standard" (Q8_0) or "small" (Q4_K_M).
    pub cleanup_size: String,
    /// A Moroccan Darija model id used instead of the Phonon choice. Lives only here: preferences.json and
    /// models\settings.json are shared with the other Hamsa, which doesn't know these ids.
    pub speech_override: Option<String>,
    /// The first-run setup has been finished or closed (or the user already had a model). Never shown again on its own.
    pub setup_done: bool,
}

impl Default for BetaSettings {
    fn default() -> Self {
        Self {
            theme: String::new(),
            microphone: String::new(),
            hands_free: true,
            smart_cleanup: false,
            cleanup_size: "standard".into(),
            speech_override: None,
            setup_done: false,
        }
    }
}

pub fn load_settings(dir: &Path) -> BetaSettings {
    let mut s: BetaSettings =
        fs::read(dir.join("beta.json")).ok().and_then(|raw| serde_json::from_slice(&raw).ok()).unwrap_or_default();
    if s.theme.is_empty() {
        s.theme = "system".into();
    }
    s
}

pub fn save_settings(dir: &Path, s: &BetaSettings) {
    save(&dir.join("beta.json"), s);
}

/// Write then rename, so a crash mid-write can't leave a half-written file.
fn save(path: &Path, value: &impl Serialize) {
    if let Ok(json) = serde_json::to_vec_pretty(value) {
        let tmp = path.with_extension("json.tmp");
        if fs::write(&tmp, json).is_ok() {
            let _ = fs::rename(&tmp, path);
        }
    }
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// One note on the Notes page. Its title is the first non-empty line of `text`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Note {
    pub id: u64,
    pub text: String,
    /// Unix time in milliseconds.
    pub created_ms: i64,
    pub updated_ms: i64,
}

/// The first non-empty line, trimmed; empty for a blank note.
pub fn note_title(text: &str) -> String {
    text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("").to_string()
}

pub fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Where a dictated `addition` goes after the existing `text`. Right after a line break it follows directly; an empty
/// note or one that is only a title line gets a new line first, so the title stays short and the speech lands in the
/// body; otherwise it follows the last word after a space.
pub fn append_text(text: &str, addition: &str) -> String {
    let addition = addition.trim();
    if addition.is_empty() {
        return text.to_string();
    }
    let sep = if text.ends_with('\n') {
        ""
    } else if !text.contains('\n') {
        "\n"
    } else if text.ends_with(' ') {
        ""
    } else {
        " "
    };
    format!("{text}{sep}{addition}")
}

/// Splits a note into the title line and the rest (the editor shows them in two fields). `join_note` is the inverse.
pub fn split_note(text: &str) -> (String, String) {
    match text.split_once('\n') {
        Some((first, rest)) => (first.trim_end_matches('\r').to_string(), rest.to_string()),
        None => (text.to_string(), String::new()),
    }
}

pub fn join_note(title: &str, body: &str) -> String {
    if body.is_empty() {
        title.to_string()
    } else {
        format!("{title}\n{body}")
    }
}

pub struct Notes {
    path: PathBuf,
    /// Newest-updated first.
    pub items: Vec<Note>,
    /// Text changed in memory but not yet written.
    dirty: bool,
}

impl Notes {
    /// A missing or unreadable file gives an empty list.
    pub fn load(dir: &Path) -> Self {
        let path = dir.join("notes.json");
        let mut items: Vec<Note> = fs::read(&path).ok().and_then(|raw| serde_json::from_slice(&raw).ok()).unwrap_or_default();
        items.sort_by(|a, b| b.updated_ms.cmp(&a.updated_ms));
        Self { path, items, dirty: false }
    }

    pub fn get(&self, id: u64) -> Option<&Note> {
        self.items.iter().find(|n| n.id == id)
    }

    /// A new empty note at the top; returns its id.
    pub fn add(&mut self, now: i64) -> u64 {
        let id = self.items.iter().map(|n| n.id + 1).max().unwrap_or(0).max(now as u64);
        self.items.insert(0, Note { id, text: String::new(), created_ms: now, updated_ms: now });
        self.write();
        id
    }

    /// Replace a note's text in memory (call `flush` to write it). Returns false when it is unchanged or missing.
    pub fn set_text(&mut self, id: u64, text: &str, now: i64) -> bool {
        let Some(pos) = self.items.iter().position(|n| n.id == id) else { return false };
        if self.items[pos].text == text {
            return false;
        }
        let mut note = self.items.remove(pos);
        note.text = text.to_string();
        note.updated_ms = now.max(note.updated_ms);
        self.items.insert(0, note);
        self.dirty = true;
        true
    }

    /// Write pending text changes to disk.
    pub fn flush(&mut self) {
        if self.dirty {
            self.write();
        }
    }

    pub fn remove(&mut self, id: u64) {
        self.items.retain(|n| n.id != id);
        self.write();
    }

    fn write(&mut self) {
        save(&self.path, &self.items);
        self.dirty = false;
    }
}
/// The Style page's formatting switches, in `style.json` (not in preferences.json, which the other Hamsa also writes).
pub fn load_formatting(dir: &Path) -> hamsa_core::Formatting {
    fs::read(dir.join("style.json")).ok().and_then(|raw| serde_json::from_slice(&raw).ok()).unwrap_or_default()
}

pub fn save_formatting(dir: &Path, f: &hamsa_core::Formatting) {
    save(&dir.join("style.json"), f);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hands_free_is_on_unless_the_file_says_otherwise() {
        let dir = temp_dir("handsfree");
        assert!(load_settings(&dir).hands_free, "no file");
        fs::write(dir.join("beta.json"), br#"{"theme":"dark","microphone":""}"#).unwrap();
        assert!(load_settings(&dir).hands_free, "older file without the key");
        let mut s = load_settings(&dir);
        s.hands_free = false;
        save_settings(&dir, &s);
        let back = load_settings(&dir);
        assert!(!back.hands_free);
        assert_eq!(back.theme, "dark");
    }

    #[test]
    fn style_without_list_keys_gets_the_list_defaults() {
        let dir = temp_dir("style-lists");
        fs::write(dir.join("style.json"), br#"{"capitalize":false,"end_punctuation":true,"remove_fillers":true,"smart_punctuation":false}"#).unwrap();
        let old = load_formatting(&dir);
        assert!(!old.capitalize, "saved values are kept");
        assert!(old.lists);
        assert_eq!(old.bullet, hamsa_core::Bullet::Dot);
        let mut f = old;
        f.lists = false;
        f.bullet = hamsa_core::Bullet::Dash;
        save_formatting(&dir, &f);
        assert_eq!(load_formatting(&dir), f);
    }

    #[test]
    fn smart_cleanup_is_off_and_standard_unless_saved() {
        let dir = temp_dir("cleanup");
        let fresh = load_settings(&dir);
        assert!(!fresh.smart_cleanup, "no file");
        assert_eq!(fresh.cleanup_size, "standard");
        fs::write(dir.join("beta.json"), br#"{"theme":"dark","hands_free":false}"#).unwrap();
        let old = load_settings(&dir);
        assert!(!old.smart_cleanup, "older file without the keys");
        assert_eq!(old.cleanup_size, "standard");
        assert!(!old.hands_free);
        let mut s = old;
        s.smart_cleanup = true;
        s.cleanup_size = "small".into();
        save_settings(&dir, &s);
        let raw = fs::read_to_string(dir.join("beta.json")).unwrap();
        assert!(raw.contains("\"smart_cleanup\": true") && raw.contains("\"cleanup_size\": \"small\""));
        let back = load_settings(&dir);
        assert!(back.smart_cleanup);
        assert_eq!(back.cleanup_size, "small");
        assert_eq!(back.theme, "dark");
        assert!(!back.hands_free);
    }

    #[test]
    fn setup_done_is_off_until_saved_and_survives_a_restart() {
        let dir = temp_dir("setup-done");
        assert!(!load_settings(&dir).setup_done, "no file");
        fs::write(dir.join("beta.json"), br#"{"theme":"dark","hands_free":false}"#).unwrap();
        assert!(!load_settings(&dir).setup_done, "older file without the key");
        let mut s = load_settings(&dir);
        s.setup_done = true;
        save_settings(&dir, &s);
        let raw = fs::read_to_string(dir.join("beta.json")).unwrap();
        assert!(raw.contains("\"setup_done\": true"));
        let back = load_settings(&dir);
        assert!(back.setup_done);
        assert_eq!(back.theme, "dark");
        assert!(!back.hands_free, "other keys are kept");
        assert!(!dir.join("preferences.json").exists(), "nothing goes into the shared preferences");
    }

    #[test]
    fn the_darija_choice_is_beta_only_and_survives_a_restart() {
        let dir = temp_dir("override");
        assert_eq!(load_settings(&dir).speech_override, None, "no file");
        fs::write(dir.join("beta.json"), br#"{"theme":"dark"}"#).unwrap();
        assert_eq!(load_settings(&dir).speech_override, None, "older file without the key");
        let mut s = load_settings(&dir);
        s.speech_override = Some("fihria-asr-0.6b".into());
        save_settings(&dir, &s);
        assert_eq!(load_settings(&dir).speech_override.as_deref(), Some("fihria-asr-0.6b"));
        assert_eq!(load_settings(&dir).theme, "dark");
        // Choosing Phonon again clears it.
        let mut s = load_settings(&dir);
        s.speech_override = None;
        save_settings(&dir, &s);
        assert_eq!(load_settings(&dir).speech_override, None);
        // Nothing was written to the shared files.
        assert!(!dir.join("preferences.json").exists() && !dir.join("models").exists());
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hamsa-notes-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_file_is_an_empty_list() {
        let dir = temp_dir("missing");
        assert!(Notes::load(&dir).items.is_empty());
    }

    #[test]
    fn corrupt_file_is_an_empty_list() {
        let dir = temp_dir("corrupt");
        fs::write(dir.join("notes.json"), b"{ not json at all").unwrap();
        assert!(Notes::load(&dir).items.is_empty());
        fs::write(dir.join("notes.json"), b"[{\"id\": \"wrong\"}]").unwrap();
        assert!(Notes::load(&dir).items.is_empty());
    }

    #[test]
    fn round_trip_keeps_notes_and_orders_by_update() {
        let dir = temp_dir("roundtrip");
        let mut n = Notes::load(&dir);
        let a = n.add(1_000);
        let b = n.add(2_000);
        assert_ne!(a, b);
        assert!(n.set_text(a, "Groceries\nmilk", 3_000));
        assert!(!n.set_text(a, "Groceries\nmilk", 4_000), "same text changes nothing");
        assert!(Notes::load(&dir).items.iter().all(|x| x.text.is_empty()), "not written before flush");
        n.flush();
        let back = Notes::load(&dir);
        assert_eq!(back.items.len(), 2);
        assert_eq!(back.items[0].id, a);
        assert_eq!(back.items[0].text, "Groceries\nmilk");
        assert_eq!(back.items[0].created_ms, 1_000);
        assert_eq!(back.items[0].updated_ms, 3_000);
        assert_eq!(back.items[1].id, b);
        assert!(!dir.join("notes.json.tmp").exists());
    }

    #[test]
    fn remove_deletes_the_note() {
        let dir = temp_dir("remove");
        let mut n = Notes::load(&dir);
        let a = n.add(1);
        n.remove(a);
        assert!(Notes::load(&dir).items.is_empty());
        assert!(n.get(a).is_none());
    }

    #[test]
    fn ids_stay_unique_for_notes_made_in_the_same_millisecond() {
        let dir = temp_dir("ids");
        let mut n = Notes::load(&dir);
        let a = n.add(5_000);
        let b = n.add(5_000);
        assert_ne!(a, b);
    }

    #[test]
    fn title_is_the_first_non_empty_line() {
        assert_eq!(note_title("Product sync\nnotes"), "Product sync");
        assert_eq!(note_title("\n  \n  Trip ideas  \nmore"), "Trip ideas");
        assert_eq!(note_title(""), "");
        assert_eq!(note_title(" \n\t"), "");
    }

    #[test]
    fn words_are_counted_across_lines() {
        assert_eq!(word_count(""), 0);
        assert_eq!(word_count("one two\nthree   four\n\nfive"), 5);
    }

    #[test]
    fn dictation_is_appended_with_the_right_separator() {
        assert_eq!(append_text("", "hello there"), "\nhello there");
        assert_eq!(append_text("Groceries", "milk"), "Groceries\nmilk");
        assert_eq!(append_text("Groceries\nmilk.", " and more "), "Groceries\nmilk. and more");
        assert_eq!(append_text("Groceries\nmilk. ", "eggs"), "Groceries\nmilk. eggs");
        assert_eq!(append_text("Hello.\n", "next"), "Hello.\nnext");
        assert_eq!(append_text("Hello", "  "), "Hello");
    }

    #[test]
    fn title_and_body_split_and_join() {
        assert_eq!(split_note("Title\nline one\nline two"), ("Title".into(), "line one\nline two".into()));
        assert_eq!(split_note("Only a title"), ("Only a title".into(), "".into()));
        assert_eq!(split_note("\nbody only"), ("".into(), "body only".into()));
        assert_eq!(join_note("Title", ""), "Title");
        assert_eq!(join_note("Title", "body"), "Title\nbody");
        for t in ["Title\n\nbody text\n", "\nbody only", "Title", ""] {
            let (a, b) = split_note(t);
            assert_eq!(join_note(&a, &b), t);
        }
    }
}
