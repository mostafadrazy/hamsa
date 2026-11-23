//! User preferences kept on the Rust side (`preferences.json` in the app data folder).

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// Hold-to-talk shortcut as key ids (see hotkey.rs). Default: Ctrl + Win.
    pub shortcut: Vec<u16>,
    /// "auto" or an ISO 639-1 code; only used with the multilingual model.
    pub language: String,
    /// Dictionary and snippets.
    pub vocabulary: crate::vocabulary::Vocabulary,
    /// Dictation bar: shown when idle, live preview, where it sits.
    pub bar: crate::layout::Layout,
    /// Fields this build doesn't know (e.g. written by another Hamsa version), kept as-is when saving.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            shortcut: vec![0x11, 0x5B],
            language: "auto".into(),
            vocabulary: Default::default(),
            bar: Default::default(),
            extra: serde_json::Map::new(),
        }
    }
}

/// Read preferences; a field that doesn't parse falls back to its default instead of losing the whole file.
fn parse(raw: &[u8]) -> Prefs {
    if let Ok(p) = serde_json::from_slice::<Prefs>(raw) {
        return p;
    }
    let Ok(serde_json::Value::Object(mut map)) = serde_json::from_slice::<serde_json::Value>(raw) else {
        return Prefs::default();
    };
    let mut prefs = Prefs::default();
    macro_rules! field {
        ($name:ident) => {
            if let Some(v) = map.remove(stringify!($name)) {
                match serde_json::from_value(v.clone()) {
                    Ok(x) => prefs.$name = x,
                    // Keep what we couldn't read, so saving doesn't erase it.
                    Err(_) => {
                        prefs.extra.insert(format!("{}_unreadable", stringify!($name)), v);
                    }
                }
            }
        };
    }
    field!(shortcut);
    field!(language);
    field!(vocabulary);
    field!(bar);
    prefs.extra.extend(map);
    prefs
}

pub struct PrefsStore {
    path: PathBuf,
    prefs: Mutex<Prefs>,
}

impl PrefsStore {
    pub fn load(data_dir: &std::path::Path) -> Self {
        let path = data_dir.join("preferences.json");
        let prefs = fs::read(&path).ok().map(|raw| parse(&raw)).unwrap_or_default();
        Self { path, prefs: Mutex::new(prefs) }
    }

    pub fn get(&self) -> Prefs {
        self.prefs.lock().map(|p| p.clone()).unwrap_or_default()
    }

    pub fn update(&self, change: impl FnOnce(&mut Prefs)) {
        if let Ok(mut p) = self.prefs.lock() {
            change(&mut p);
            if let Some(dir) = self.path.parent() {
                let _ = fs::create_dir_all(dir);
            }
            if let Ok(json) = serde_json::to_vec_pretty(&*p) {
                // Write then rename, so a crash mid-write can't leave a half-written file.
                let tmp = self.path.with_extension("json.tmp");
                if fs::write(&tmp, json).is_ok() {
                    let _ = fs::rename(&tmp, &self.path);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_unknown_fields_and_reads_shared_shape() {
        let raw = br#"{"shortcut":[119],"language":"el","save_audio":true,
            "vocabulary":{"dictionary":[{"spoken":"a","text":"A"}],"snippets":[]},
            "bar":{"visible":false,"preview":true,"placement":{"mode":"custom","x":-5,"y":10},"monitor":"X"}}"#;
        let p = parse(raw);
        assert_eq!(p.shortcut, vec![119]);
        assert_eq!(p.vocabulary.dictionary.len(), 1);
        assert!(!p.bar.visible);
        let back = serde_json::to_value(&p).unwrap();
        assert_eq!(back["save_audio"], true);
        assert_eq!(back["bar"]["placement"]["mode"], "custom");
        assert_eq!(back["bar"]["placement"]["x"], -5);
    }

    #[test]
    fn a_bad_field_does_not_lose_the_rest() {
        let p = parse(br#"{"shortcut":[119],"bar":{"placement":{"mode":"sideways"}}}"#);
        assert_eq!(p.shortcut, vec![119]);
        assert!(p.bar.visible);
        assert!(p.extra.contains_key("bar_unreadable"));
    }
}
