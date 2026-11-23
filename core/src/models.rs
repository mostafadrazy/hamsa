//! Speech models: download (with progress and a pinned checksum), unpack, load, and pick the active one.
//!
//! Phonon models live in `%LOCALAPPDATA%\app.hamsa.desktop\models\<id>\`; the choice is saved in `settings.json`
//! next to it (shared with the other Hamsa, so it only ever holds Phonon ids).
//!
//! The Moroccan Darija models (`llm::asr`) run on the llama.cpp helper and live in `llm\<id>\`. Choosing one is an
//! in-memory override of the Phonon choice, "use Darija instead"; the app saves it in its own file (beta.json) and
//! hands it back at startup with `set_speech_override`. Nothing about it is written to the shared files.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde::Serialize;
use sha2::{Digest, Sha256};
use crate::llm::asr::{Asr, AsrModel};
use crate::llm::{HostState, LlamaHost};
use crate::{Event, Sink};

pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    url: &'static str,
    sha256: &'static str,
    pub bytes: u64,
}

/// Fermion Research's release archives on Hugging Face, pinned by SHA-256.
pub const CATALOG: [ModelInfo; 2] = [
    ModelInfo {
        id: "phonon-2.5-en",
        name: "Phonon-2.5 English",
        url: "https://huggingface.co/FermionResearch/Phonon-2.5-English/resolve/main/phonon-2.5-en.bps.tar.zst",
        sha256: "6bd8119ac5a637e5ece0666f5c1e43dbcd99e1c36b3d0ef3ec5f9b58260a9dd4",
        bytes: 163_693_770,
    },
    ModelInfo {
        id: "phonon-2.5",
        name: "Phonon-2.5",
        url: "https://huggingface.co/FermionResearch/Phonon-2.5/resolve/main/phonon-2.5.bps.tar.zst",
        sha256: "1faa04053a472cabab3bc31d07f30c23f22906f71044ca10eb1f0de36d1f19a9",
        bytes: 162_587_031,
    },
];

fn info(id: &str) -> Option<&'static ModelInfo> {
    CATALOG.iter().find(|m| m.id == id)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub id: &'static str,
    pub name: &'static str,
    pub bytes: u64,
    pub installed: bool,
    pub active: bool,
    pub loading: bool,
    /// 0–1 while downloading.
    pub progress: Option<f32>,
    pub error: Option<String>,
    /// Where the loaded model runs ("GPU · name" or "CPU"), for the active model only.
    pub device: Option<String>,
    /// Why the GPU isn't used although it was allowed (active model only).
    pub device_note: Option<String>,
    /// Can be deleted from the card (the Darija models; the Phonon ones are replaced, not removed).
    pub removable: bool,
}

/// What turns speech into text right now.
#[derive(Clone)]
pub enum SpeechBackend {
    /// Phonon, in this process.
    Phonon { id: String, engine: Arc<phonon::Engine> },
    /// A Darija model, on the llama.cpp helper.
    Llama(AsrModel),
}

impl SpeechBackend {
    /// The model's id (for `Finished.model` and the English-only gates).
    pub fn id(&self) -> String {
        match self {
            SpeechBackend::Phonon { id, .. } => id.clone(),
            SpeechBackend::Llama(m) => m.id().to_string(),
        }
    }
}

pub struct Models {
    sink: Sink,
    dir: PathBuf,
    active: RwLock<Option<(String, Arc<phonon::Engine>)>>,
    loading: Mutex<Option<String>>,
    progress: Mutex<HashMap<String, f32>>,
    errors: Mutex<HashMap<String, String>>,
    compute: Mutex<phonon::Compute>,
    /// The llama.cpp helper, shared by the Darija models and smart cleanup (it holds one model at a time).
    host: Arc<LlamaHost>,
    asr: Arc<Asr>,
    /// When set, this Darija model is used instead of the chosen Phonon model.
    speech_override: RwLock<Option<AsrModel>>,
    /// True while a Darija model is in use; smart cleanup (English only) stays away from the helper then.
    llm_active: Arc<AtomicBool>,
}

/// "auto" | "gpu" | "cpu" ⇄ the engine's setting.
pub fn compute_from(name: &str) -> Option<phonon::Compute> {
    match name {
        "auto" => Some(phonon::Compute::Auto),
        "gpu" => Some(phonon::Compute::Gpu),
        "cpu" => Some(phonon::Compute::Cpu),
        _ => None,
    }
}

pub fn compute_name(c: phonon::Compute) -> &'static str {
    match c {
        phonon::Compute::Auto => "auto",
        phonon::Compute::Gpu => "gpu",
        phonon::Compute::Cpu => "cpu",
    }
}

impl Models {
    pub fn new(data_dir: &std::path::Path, sink: Sink) -> Result<Arc<Self>, String> {
        let dir = data_dir.join("models");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        // Kept in its own file so another Hamsa build sharing preferences.json can't drop it.
        let compute = fs::read(dir.join("compute.json"))
            .ok()
            .and_then(|raw| serde_json::from_slice::<serde_json::Value>(&raw).ok())
            .and_then(|v| v["compute"].as_str().and_then(compute_from))
            .unwrap_or(phonon::Compute::Auto);
        let host = LlamaHost::new(data_dir);
        let me = Arc::new(Self {
            sink,
            dir,
            active: RwLock::new(None),
            loading: Mutex::new(None),
            progress: Mutex::new(HashMap::new()),
            errors: Mutex::new(HashMap::new()),
            compute: Mutex::new(compute),
            asr: Asr::new(data_dir, host.clone()),
            host,
            speech_override: RwLock::new(None),
            llm_active: Arc::new(AtomicBool::new(false)),
        });
        // The Darija cards follow the helper (loading, ready on GPU or CPU, failed).
        let weak = Arc::downgrade(&me);
        me.host.on_change(Arc::new(move || {
            if let Some(m) = weak.upgrade() {
                m.changed();
            }
        }));
        Ok(me)
    }

    /// The llama.cpp helper, for smart cleanup to share.
    pub fn llama_host(&self) -> Arc<LlamaHost> {
        self.host.clone()
    }

    /// Raised while a Darija model is in use (see `Cleanup::new`).
    pub fn llm_active_flag(&self) -> Arc<AtomicBool> {
        self.llm_active.clone()
    }

    pub fn asr(&self) -> &Arc<Asr> {
        &self.asr
    }

    /// The Darija model used instead of the Phonon choice, if any.
    pub fn speech_override(&self) -> Option<AsrModel> {
        self.speech_override.read().ok().and_then(|o| *o)
    }

    fn set_override(&self, m: Option<AsrModel>) {
        if let Ok(mut o) = self.speech_override.write() {
            *o = m;
        }
        self.llm_active.store(m.is_some(), Ordering::SeqCst);
    }

    /// Restore the app's saved Darija choice at startup. Returns false (and uses Phonon) when `id` isn't an installed
    /// Darija model, so the app can drop it from its settings.
    pub fn set_speech_override(&self, id: Option<&str>) -> bool {
        match id.map(AsrModel::from_id) {
            None => {
                self.set_override(None);
                true
            }
            Some(Some(m)) if self.asr.installed(m) => {
                self.set_override(Some(m));
                self.changed();
                true
            }
            Some(_) => {
                self.set_override(None);
                false
            }
        }
    }

    /// What transcribes speech now: the Darija override if set, else the loaded Phonon model, else nothing.
    pub fn backend(&self) -> Option<SpeechBackend> {
        if let Some(m) = self.speech_override() {
            return Some(SpeechBackend::Llama(m));
        }
        self.engine().map(|(id, engine)| SpeechBackend::Phonon { id, engine })
    }

    /// Start loading the Darija model in use, so it is ready when the recording is. False when Phonon is in use.
    pub fn warm_up_speech(&self) -> bool {
        match self.speech_override() {
            Some(m) => {
                self.asr.warm_up(m);
                true
            }
            None => false,
        }
    }

    /// Use a Darija model instead of Phonon.
    fn use_asr(&self, m: AsrModel) {
        if !self.asr.installed(m) {
            self.set_error(m.id(), Some("This model isn't downloaded yet.".into()));
            self.changed();
            return;
        }
        self.set_error(m.id(), None);
        self.set_override(Some(m));
        // The helper holds one model at a time: this replaces smart cleanup's (or the other Darija model).
        self.asr.warm_up(m);
        self.changed();
    }

    /// Back to Phonon; frees the helper if a Darija model was loaded in it.
    fn clear_override(&self) {
        self.set_override(None);
        self.asr.stop_if_serving();
    }

    /// Delete a downloaded Darija model.
    pub fn remove(&self, id: &str) {
        let Some(m) = AsrModel::from_id(id) else { return };
        if self.speech_override() == Some(m) {
            self.clear_override();
        }
        self.asr.remove(m);
        self.set_error(id, None);
        self.changed();
    }

    pub fn dir(&self) -> &PathBuf {
        &self.dir
    }

    fn model_dir(&self, id: &str) -> PathBuf {
        self.dir.join(id)
    }

    fn installed(&self, id: &str) -> Option<PathBuf> {
        phonon::find_model_dir(&self.model_dir(id))
    }

    fn settings_path(&self) -> PathBuf {
        self.dir.join("settings.json")
    }

    fn saved_choice(&self) -> Option<String> {
        let raw = fs::read(self.settings_path()).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&raw).ok()?;
        v["active_model"].as_str().map(String::from)
    }

    fn save_choice(&self, id: &str) {
        let _ = fs::write(self.settings_path(), serde_json::json!({ "active_model": id }).to_string());
    }

    pub fn status(&self) -> Vec<ModelStatus> {
        let compute = self.compute();
        let (active, device, note) = self
            .active
            .read()
            .ok()
            .and_then(|a| {
                a.as_ref().map(|(id, e)| {
                    let note = (compute != phonon::Compute::Cpu).then(|| e.gpu_unavailable_reason().map(String::from)).flatten();
                    (Some(id.clone()), Some(e.device()), note)
                })
            })
            .unwrap_or((None, None, None));
        let loading = self.loading.lock().ok().and_then(|l| l.clone());
        let progress = self.progress.lock().map(|p| p.clone()).unwrap_or_default();
        let errors = self.errors.lock().map(|e| e.clone()).unwrap_or_default();
        let darija = self.speech_override();
        let phonon = CATALOG.iter().map(|m| ModelStatus {
                id: m.id,
                name: m.name,
                bytes: m.bytes,
                installed: self.installed(m.id).is_some(),
                active: darija.is_none() && active.as_deref() == Some(m.id),
                loading: loading.as_deref() == Some(m.id),
                progress: progress.get(m.id).copied(),
                error: errors.get(m.id).cloned(),
                device: if active.as_deref() == Some(m.id) { device.clone() } else { None },
                device_note: if active.as_deref() == Some(m.id) { note.clone() } else { None },
                removable: false,
            });
        let llama = AsrModel::ALL.into_iter().map(|m| {
            let state = self.asr.host_state(m);
            let failed = match &state {
                Some(HostState::Failed(e)) if darija == Some(m) => Some(e.clone()),
                _ => None,
            };
            ModelStatus {
                id: m.id(),
                name: m.name(),
                bytes: m.total_bytes(),
                installed: self.asr.installed(m),
                active: darija == Some(m),
                loading: state == Some(HostState::Starting),
                progress: progress.get(m.id()).copied(),
                error: errors.get(m.id()).cloned().or(failed),
                device: match state {
                    Some(HostState::Ready(b)) => Some(b.label().to_string()),
                    _ => None,
                },
                device_note: None,
                removable: true,
            }
        });
        phonon.chain(llama).collect()
    }

    fn changed(&self) {
        (self.sink)(Event::Models(self.status()));
    }

    fn set_error(&self, id: &str, msg: Option<String>) {
        if let Ok(mut e) = self.errors.lock() {
            match msg {
                Some(m) => {
                    eprintln!("hamsa: model {id}: {m}");
                    e.insert(id.to_string(), m);
                }
                None => {
                    e.remove(id);
                }
            }
        }
    }

    pub fn compute(&self) -> phonon::Compute {
        self.compute.lock().map(|c| *c).unwrap_or(phonon::Compute::Auto)
    }

    /// Choose where Phonon runs, and reload the active model to apply it.
    pub fn set_compute(self: &Arc<Self>, compute: phonon::Compute) {
        if let Ok(mut c) = self.compute.lock() {
            if *c == compute {
                return;
            }
            *c = compute;
        }
        let _ = fs::write(self.dir.join("compute.json"), serde_json::json!({ "compute": compute_name(compute) }).to_string());
        if let Some((id, _)) = self.engine() {
            self.load(id);
        } else {
            self.changed();
        }
    }

    /// The loaded engine, if any.
    pub fn engine(&self) -> Option<(String, Arc<phonon::Engine>)> {
        self.active.read().ok().and_then(|a| a.clone())
    }

    /// Load the saved choice (or the first installed model) in the background.
    pub fn startup(self: &Arc<Self>) {
        let choice = self
            .saved_choice()
            .filter(|id| self.installed(id).is_some())
            .or_else(|| CATALOG.iter().find(|m| self.installed(m.id).is_some()).map(|m| m.id.to_string()));
        if let Some(id) = choice {
            self.load(id);
        }
    }

    /// The user picked `id` (a Phonon or a Darija model): make it the one that transcribes. Choosing a Phonon model
    /// ends a Darija override; the app saves `speech_override()` afterwards.
    pub fn activate(self: &Arc<Self>, id: String) {
        if let Some(m) = AsrModel::from_id(&id) {
            self.use_asr(m);
            return;
        }
        if info(&id).is_some() && self.installed(&id).is_some() && self.speech_override().is_some() {
            self.clear_override();
        }
        self.load(id);
    }

    /// Load the Phonon model `id` in the background and remember it as the Phonon choice.
    fn load(self: &Arc<Self>, id: String) {
        if info(&id).is_none() {
            return;
        }
        let Some(dir) = self.installed(&id) else {
            self.set_error(&id, Some("This model isn't downloaded yet.".into()));
            self.changed();
            return;
        };
        if let Ok(mut l) = self.loading.lock() {
            *l = Some(id.clone());
        }
        self.set_error(&id, None);
        self.changed();
        let me = self.clone();
        let compute = self.compute();
        std::thread::spawn(move || {
            match phonon::Engine::load_with(&dir, compute) {
                Ok(engine) => {
                    if let Ok(mut a) = me.active.write() {
                        *a = Some((id.clone(), Arc::new(engine)));
                    }
                    me.save_choice(&id);
                }
                Err(e) => me.set_error(&id, Some(format!("Couldn't load the model: {e}"))),
            }
            if let Ok(mut l) = me.loading.lock() {
                *l = None;
            }
            me.changed();
        });
    }

    /// Download, verify and unpack `id`, then make it active.
    pub fn download(self: &Arc<Self>, id: String) {
        if let Some(m) = AsrModel::from_id(&id) {
            self.download_asr(m);
            return;
        }
        let Some(model) = info(&id) else { return };
        if let Ok(mut p) = self.progress.lock() {
            if p.contains_key(&id) {
                return; // already downloading
            }
            p.insert(id.clone(), 0.0);
        }
        self.set_error(&id, None);
        self.changed();
        let me = self.clone();
        std::thread::spawn(move || {
            let result = me.fetch_and_unpack(model);
            if let Ok(mut p) = me.progress.lock() {
                p.remove(model.id);
            }
            match result {
                Ok(()) => me.load(model.id.to_string()),
                Err(e) => {
                    me.set_error(model.id, Some(e));
                    me.changed();
                }
            }
        });
    }

    /// Download llama.cpp (if missing) and both files of a Darija model. It is not switched on afterwards: the card's
    /// "Use this model" does that.
    fn download_asr(self: &Arc<Self>, m: AsrModel) {
        if let Ok(mut p) = self.progress.lock() {
            if p.contains_key(m.id()) {
                return; // already downloading
            }
            p.insert(m.id().to_string(), 0.0);
        }
        self.set_error(m.id(), None);
        self.changed();
        let me = self.clone();
        std::thread::spawn(move || {
            let result = me.asr.fetch(m, &mut |p| {
                if let Ok(mut map) = me.progress.lock() {
                    map.insert(m.id().to_string(), p);
                }
                me.changed();
            });
            if let Ok(mut p) = me.progress.lock() {
                p.remove(m.id());
            }
            if let Err(e) = result {
                me.set_error(m.id(), Some(e));
            }
            me.changed();
        });
    }

    fn fetch_and_unpack(&self, model: &ModelInfo) -> Result<(), String> {
        let archive = self.dir.join(format!("{}.bps.tar.zst", model.id));
        let partial = archive.with_extension("part");
        self.fetch(model, &partial)?;
        fs::rename(&partial, &archive).map_err(|e| e.to_string())?;

        let target = self.model_dir(model.id);
        let _ = fs::remove_dir_all(&target);
        let unpacked = phonon::unpack(&archive, &target);
        let _ = fs::remove_file(&archive); // the unpacked model is all we keep
        unpacked.map(|_| ()).map_err(|e| {
            let _ = fs::remove_dir_all(&target);
            format!("Couldn't unpack the model: {e}")
        })
    }

    fn fetch(&self, model: &ModelInfo, path: &PathBuf) -> Result<(), String> {
        let response = ureq::get(model.url)
            .call()
            .map_err(|e| format!("Download failed: {e}. Check your internet connection."))?;
        let total = response
            .header("Content-Length")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(model.bytes)
            .max(1);
        let mut reader = response.into_reader();
        let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        let (mut received, mut last_reported) = (0u64, -1i64);
        loop {
            let n = reader.read(&mut buf).map_err(|e| format!("Download interrupted: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            hasher.update(&buf[..n]);
            received += n as u64;
            let percent = (received * 100 / total) as i64;
            if percent != last_reported {
                last_reported = percent;
                if let Ok(mut p) = self.progress.lock() {
                    p.insert(model.id.to_string(), (received as f32 / total as f32).min(1.0));
                }
                self.changed();
            }
        }
        file.flush().map_err(|e| e.to_string())?;
        drop(file);
        let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if digest != model.sha256 {
            let _ = fs::remove_file(path);
            return Err("The download was corrupted (checksum mismatch). Please try again.".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::cleanup::{Cleanup, Size};

    fn fake_install(models: &Models, m: AsrModel) {
        // Files of exactly the pinned sizes (empty inside), so the model counts as downloaded.
        fs::create_dir_all(models.asr().dir(m)).unwrap();
        for (path, part) in [(models.asr().llm_path(m), m.llm()), (models.asr().mmproj_path(m), m.mmproj())] {
            fs::File::create(path).unwrap().set_len(part.bytes).unwrap();
        }
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hamsa-models-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_darija_override_picks_the_backend_and_never_touches_the_shared_files() {
        let dir = temp("override");
        let models = Models::new(&dir, Arc::new(|_| {})).unwrap();
        assert!(models.backend().is_none(), "nothing installed, nothing loaded");
        // A saved choice for a model that isn't downloaded is refused (the app then drops it from beta.json).
        assert!(!models.set_speech_override(Some("fihria-asr-0.6b")));
        assert!(models.speech_override().is_none());
        assert!(!models.set_speech_override(Some("phonon-2.5")), "not a Darija id");
        fake_install(&models, AsrModel::Fast);
        assert!(models.set_speech_override(Some("fihria-asr-0.6b")));
        assert!(matches!(models.backend(), Some(SpeechBackend::Llama(AsrModel::Fast))));
        assert_eq!(models.backend().unwrap().id(), "fihria-asr-0.6b");
        assert!(models.llm_active_flag().load(Ordering::SeqCst));
        // The cards: only the Darija one is active; the Phonon cards are not.
        let list = models.status();
        assert_eq!(list.len(), 4);
        assert!(list.iter().filter(|m| m.active).map(|m| m.id).eq(["fihria-asr-0.6b"]));
        assert!(list.iter().filter(|m| m.removable).all(|m| m.id.starts_with("fihria-asr-")));
        // Choosing a Phonon model that isn't there changes nothing; and nothing was written to models\settings.json.
        models.activate("phonon-2.5".into());
        assert!(models.speech_override().is_some());
        assert!(!models.dir().join("settings.json").exists(), "the shared choice file is untouched");
        // Removing the model ends the override.
        models.remove("fihria-asr-0.6b");
        assert!(models.speech_override().is_none());
        assert!(!models.llm_active_flag().load(Ordering::SeqCst));
        assert!(!models.asr().installed(AsrModel::Fast));
        assert!(models.set_speech_override(None));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn smart_cleanup_stays_out_while_a_darija_model_is_in_use() {
        let dir = temp("cleanup-skip");
        let models = Models::new(&dir, Arc::new(|_| {})).unwrap();
        let cleanup = Cleanup::new(&dir, Arc::new(|_| {}), models.llama_host(), models.llm_active_flag());
        let model = dir.join("llm").join("speakoflow-mini");
        fs::create_dir_all(&model).unwrap();
        fs::File::create(model.join("SpeakoFlow-Mini-0.8B-Q4_K_M.gguf")).unwrap().set_len(Size::Small.bytes()).unwrap();
        cleanup.configure(true, Size::Small);
        assert!(cleanup.active(), "on and downloaded");
        fake_install(&models, AsrModel::Fast);
        assert!(models.set_speech_override(Some("fihria-asr-0.6b")));
        assert!(!cleanup.active(), "off the helper while Darija is in use, even though the switch is on");
        models.remove("fihria-asr-0.6b");
        assert!(cleanup.active(), "back once Phonon is in use again");
        let _ = fs::remove_dir_all(&dir);
    }
}
