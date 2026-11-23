//! Wires the core (dictation, models, shortcut, preferences) to the two Slint windows.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use hamsa_core::{dictation::Control, models, BarPhase, DictationHandle, Event, Models, PrefsStore, Rule};
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};

use crate::bar::Bar;
use crate::flow;
use crate::store::{self, BetaSettings, Dictation, History, Notes};
use crate::{win, AppState, BarWindow, DictEntry, HistoryGroup, HistoryItem, MainWindow, ModelCard, NoteEntry, SetupWindow, SnippetEntry, Theme};

mod setup;

const MULTILINGUAL: &str = "phonon-2.5";

pub const LANGUAGES: [(&str, &str); 25] = [
    ("en", "English"), ("es", "Spanish"), ("fr", "French"), ("de", "German"), ("it", "Italian"),
    ("pt", "Portuguese"), ("nl", "Dutch"), ("pl", "Polish"), ("ru", "Russian"), ("uk", "Ukrainian"),
    ("ro", "Romanian"), ("cs", "Czech"), ("sk", "Slovak"), ("bg", "Bulgarian"), ("hr", "Croatian"),
    ("sv", "Swedish"), ("da", "Danish"), ("fi", "Finnish"), ("et", "Estonian"), ("lv", "Latvian"),
    ("lt", "Lithuanian"), ("hu", "Hungarian"), ("sl", "Slovenian"), ("mt", "Maltese"), ("el", "Greek"),
];

/// Card text for each model.
fn about(id: &str) -> (&'static str, i32, &'static str, &'static str) {
    match id {
        "phonon-2.5-en" => ("Faster", 1, "English only. Lowest latency, tuned for everyday dictation into any app.", "English"),
        "fihria-asr-0.6b" => (
            "Darija",
            0,
            "Moroccan Darija (Arabic script, keeps French/English words). The faster of the two, a good fit for most computers.",
            "Moroccan Darija",
        ),
        "fihria-asr-1.7b" => (
            "Darija",
            0,
            "Moroccan Darija (Arabic script, keeps French/English words). More accurate, larger and slower, best with a graphics card.",
            "Moroccan Darija",
        ),
        _ => (
            "Multilingual base",
            2,
            "The base model. Detects the language you speak and transcribes many languages, a little slower.",
            "Multilingual, auto-detect",
        ),
    }
}

pub struct App {
    pub main: MainWindow,
    pub bar: Rc<Bar>,
    data_dir: PathBuf,
    prefs: Arc<PrefsStore>,
    models: Arc<Models>,
    dictation: DictationHandle,
    history: RefCell<History>,
    settings: RefCell<BetaSettings>,
    microphones: RefCell<Vec<String>>,
    captured: RefCell<Vec<u16>>,
    /// Dictionary search text and filter (0 all, 1 words, 2 replacements); width the word chips wrap to.
    dict_view: RefCell<(String, i32)>,
    chip_avail: Cell<f32>,
    snippet_query: RefCell<String>,
    notes: RefCell<Notes>,
    note_query: RefCell<String>,
    /// The note shown in the Notes page's editor.
    open_note: Cell<Option<u64>>,
    /// Saves the open note a moment after typing stops.
    save_timer: slint::Timer,
    /// The first-run setup window and what it keeps while open.
    setup: setup::Setup,
}

/// A snippet as the list shows it: the first line of its text stands in for the whole.
fn snippet_entry(index: usize, r: &Rule) -> SnippetEntry {
    SnippetEntry {
        spoken: r.spoken.as_str().into(),
        text: r.text.as_str().into(),
        first_line: r.text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim().into(),
        index: index as i32,
    }
}

/// "164 MB" or "1.84 GB".
fn size_label(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

fn strings(v: impl IntoIterator<Item = String>) -> ModelRc<SharedString> {
    let v: Vec<SharedString> = v.into_iter().map(SharedString::from).collect();
    ModelRc::new(VecModel::from(v))
}

fn labels(keys: &[u16]) -> Vec<String> {
    keys.iter().map(|&k| hamsa_core::hotkey::label(k)).collect()
}

impl App {
    pub fn start() -> Result<Rc<Self>, Box<dyn std::error::Error>> {
        let data_dir = hamsa_core::data_dir();
        let prefs = Arc::new(PrefsStore::load(&data_dir));
        let main = MainWindow::new()?;
        let bar_win = BarWindow::new()?;
        let setup_win = SetupWindow::new()?;

        // Core events arrive on background threads; handle them on the UI thread.
        let app_cell: Rc<RefCell<Option<Rc<App>>>> = Rc::new(RefCell::new(None));
        let main_weak = main.as_weak();
        let sink: hamsa_core::Sink = Arc::new(move |ev: Event| {
            let main_weak = main_weak.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if main_weak.upgrade().is_some() {
                    APP.with(|a| {
                        if let Some(app) = a.borrow().as_ref() {
                            app.handle(ev);
                        }
                    });
                }
            });
        });

        let models = Models::new(&data_dir, sink.clone())?;
        let settings = store::load_settings(&data_dir);
        // The Darija choice is Beta-only (beta.json); a model that is no longer downloaded is dropped.
        let mut settings = settings;
        if settings.speech_override.is_some() && !models.set_speech_override(settings.speech_override.as_deref()) {
            settings.speech_override = None;
            store::save_settings(&data_dir, &settings);
        }
        let dictation = hamsa_core::dictation::spawn(sink.clone(), data_dir.clone(), models.clone(), prefs.clone());

        // Hold-to-talk shortcut; fall back to Ctrl + Win if the saved one can't be used.
        let mut keys = prefs.get().shortcut;
        if hamsa_core::hotkey::validate(&keys).is_err() {
            keys = hamsa_core::Prefs::default().shortcut;
        }
        let capture_sink = sink.clone();
        hamsa_core::hotkey::start(dictation.tx.clone(), &keys, move |raw| {
            capture_sink(Event::ShortcutCaptured(hamsa_core::hotkey::normalize(&raw)));
        });

        let history = History::load(&data_dir);
        let notes = Notes::load(&data_dir);
        let bar = Bar::new(bar_win, prefs.clone());

        let app = Rc::new(App {
            main,
            bar,
            data_dir,
            prefs,
            models: models.clone(),
            dictation,
            history: RefCell::new(history),
            settings: RefCell::new(settings),
            microphones: RefCell::new(Vec::new()),
            captured: RefCell::new(Vec::new()),
            dict_view: RefCell::new((String::new(), 0)),
            chip_avail: Cell::new(0.0),
            snippet_query: RefCell::new(String::new()),
            notes: RefCell::new(notes),
            note_query: RefCell::new(String::new()),
            open_note: Cell::new(None),
            save_timer: slint::Timer::default(),
            setup: setup::Setup::new(setup_win),
        });
        APP.with(|a| *a.borrow_mut() = Some(app.clone()));
        drop(app_cell);

        app.apply_theme();
        app.init_formatting();
        app.init_hands_free();
        app.init_autostart();
        app.init_cleanup();
        app.refresh_shortcut();
        app.refresh_microphones();
        app.refresh_vocabulary();
        app.refresh_history();
        app.refresh_bar_settings();
        app.init_languages();
        app.refresh_models(&models.status());
        app.connect();
        app.claim_dictation_for_notes();
        models.startup();
        app.bar.start();
        Ok(app)
    }

    /// While you are in a note in Hamsa's own window, dictated text goes into the note; everywhere else it is pasted.
    fn claim_dictation_for_notes(&self) {
        let deliver: hamsa_core::Deliver = Arc::new(|text: &str| {
            let (tx, rx) = std::sync::mpsc::channel::<bool>();
            let abandoned = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = abandoned.clone();
            let text = text.to_string();
            let queued = slint::invoke_from_event_loop(move || {
                let taken = !flag.load(std::sync::atomic::Ordering::SeqCst)
                    && APP.with(|a| a.borrow().as_ref().is_some_and(|app| app.deliver_to_setup(&text) || app.deliver_to_note(&text)));
                let _ = tx.send(taken);
            });
            // If the window can't answer in time, paste as usual.
            queued.is_ok()
                && rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap_or_else(|_| {
                    abandoned.store(true, std::sync::atomic::Ordering::SeqCst);
                    false
                })
        });
        if let Ok(mut d) = self.dictation.deliver.lock() {
            *d = Some(deliver);
        }
    }

    /// Load the Style page's switches, show them and give them to the dictation engine.
    fn init_formatting(&self) {
        let f = store::load_formatting(&self.data_dir);
        let s = self.state();
        s.set_fmt_capitalize(f.capitalize);
        s.set_fmt_end(f.end_punctuation);
        s.set_fmt_fillers(f.remove_fillers);
        s.set_fmt_smart(f.smart_punctuation);
        s.set_fmt_lists(f.lists);
        s.set_fmt_bullet(if f.bullet == hamsa_core::Bullet::Dash { 1 } else { 0 });
        if let Ok(mut cur) = self.dictation.formatting.lock() {
            *cur = f;
        }
    }

    /// Load the hands-free switch (Beta-only, in beta.json) and give it to the dictation engine and the bar.
    fn init_hands_free(&self) {
        let on = self.settings.borrow().hands_free;
        self.dictation.hands_free.store(on, std::sync::atomic::Ordering::SeqCst);
        self.state().set_hands_free(on);
        self.bar.win.set_hands_free(on);
    }

    /// Load smart cleanup's switch and size (Beta-only, in beta.json), give them to the core and show them.
    /// The Run key is the source of truth for the switch; if it is on but the exe moved, the value is rewritten.
    fn init_autostart(&self) {
        crate::autostart::refresh_path();
        self.state().set_start_with_windows(crate::autostart::is_enabled());
    }

    fn init_cleanup(&self) {
        let (on, size) = {
            let s = self.settings.borrow();
            (s.smart_cleanup, hamsa_core::CleanupSize::from_name(&s.cleanup_size))
        };
        let cleanup = &self.dictation.cleanup;
        cleanup.configure(on, size);
        self.state().set_cleanup_on(on);
        self.setup.state().set_cleanup_on(on);
        self.state().set_cleanup_size(if size == hamsa_core::CleanupSize::Small { 1 } else { 0 });
        self.refresh_cleanup(&cleanup.status());
    }

    /// The Smart cleanup card's status line, progress and buttons.
    fn refresh_cleanup(&self, st: &hamsa_core::CleanupStatus) {
        use hamsa_core::LlmState;
        let (text, kind, progress) = match &st.state {
            LlmState::NotDownloaded => ("Not downloaded".to_string(), 0, 0.0),
            LlmState::Downloading(p) => (format!("Downloading {} %", (p * 100.0).round()), 1, *p),
            LlmState::Installed if st.enabled => ("Downloaded · loads when you dictate".to_string(), 2, 0.0),
            LlmState::Installed => ("Downloaded".to_string(), 2, 0.0),
            LlmState::Starting => ("Loading…".to_string(), 2, 0.0),
            LlmState::Ready(b) => (format!("Ready on {}", b.label()), 2, 0.0),
            LlmState::Error(e) => (format!("Error: {e}"), 3, 0.0),
        };
        let s = self.state();
        s.set_cleanup_status(text.into());
        s.set_cleanup_state(kind);
        s.set_cleanup_progress(progress);
        self.refresh_setup_cleanup(st);
    }

    fn state(&self) -> AppState<'_> {
        self.main.global::<AppState>()
    }

    // ---------- Events from the core ----------

    fn handle(&self, ev: Event) {
        match ev {
            Event::Bar { phase, level } => {
                self.setup_level(level);
                self.bar.set_phase(phase, level)
            }
            Event::Preview { stable, tentative } => self.bar.set_preview(&stable, &tentative),
            Event::Locked(on) => self.bar.set_locked(on),
            Event::Models(list) => self.refresh_models(&list),
            Event::Cleanup(status) => self.refresh_cleanup(&status),
            Event::Finished(f) => self.finished(f),
            Event::Error(msg) => {
                self.setup_error(&msg);
                let s = self.state();
                s.set_status_kind(2);
                s.set_status_title("Something went wrong".into());
                s.set_status_body(msg.into());
                s.set_status_get_model(false);
            }
            Event::ShortcutCaptured(keys) => {
                let s = self.state();
                if keys.is_empty() {
                    s.set_capture_mode(0); // Esc
                    return;
                }
                let error = hamsa_core::hotkey::validate(&keys).err().unwrap_or_default();
                s.set_captured(strings(labels(&keys)));
                s.set_capture_error(error.into());
                s.set_capture_mode(2);
                *self.captured.borrow_mut() = keys;
            }
        }
        let _ = BarPhase::Idle;
    }

    fn finished(&self, f: hamsa_core::Finished) {
        // The setup window's microphone test takes its own result (and keeps it out of the history).
        if self.setup_finished(&f) {
            return;
        }
        let s = self.state();
        s.set_status_get_model(false);
        match &f.text {
            None => {
                s.set_status_kind(1);
                s.set_status_title(format!("Recorded {:.1} s, but no speech model is installed", f.audio_seconds).into());
                s.set_status_body("Download Phonon-2.5 English to turn your voice into text.".into());
                s.set_status_get_model(true);
            }
            Some(t) if t.trim().is_empty() => {
                let quiet = f.peak < 0.02;
                s.set_status_kind(1);
                s.set_status_title(if quiet { "Hamsa heard almost nothing" } else { "No words recognised" }.into());
                s.set_status_body(
                    if quiet {
                        "Check that the right microphone is selected in Settings and that Windows lets apps use it."
                    } else {
                        "Try speaking a little closer to the microphone."
                    }
                    .into(),
                );
            }
            Some(t) => {
                s.set_status_kind(1);
                s.set_status_title(t.as_str().into());
                s.set_status_body(
                    format!(
                        "{} · {:.1} s of speech in {:.2} s",
                        if f.pasted {
                            "Pasted into your app"
                        } else if f.delivered {
                            "Added to your note"
                        } else {
                            "Transcribed"
                        },
                        f.audio_seconds,
                        f.transcribe_ms as f64 / 1000.0
                    )
                    .into(),
                );
                let text = t.trim().to_string();
                let words = text.split_whitespace().count() as u32;
                self.history.borrow_mut().add(Dictation { text, at: store::now_ms(), words, duration_ms: f.duration_ms });
                self.refresh_history();
            }
        }
    }

    // ---------- Notes ----------

    /// "Today", "Yesterday" or "Nov 18" (with the year when it isn't this one).
    fn day_label(ms: i64) -> String {
        const DAY: i64 = 24 * 60 * 60 * 1000;
        const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        let now = store::now_ms();
        let date = |ms: i64| {
            let (y, m, d, _, _, _) = win::local_time(ms);
            (y, m, d)
        };
        let (y, m, d) = date(ms);
        if (y, m, d) == date(now) {
            "Today".to_string()
        } else if (y, m, d) == date(now - DAY) {
            "Yesterday".to_string()
        } else if y == date(now).0 {
            format!("{} {}", MONTHS[(m as usize + 11) % 12], d)
        } else {
            format!("{} {}, {}", MONTHS[(m as usize + 11) % 12], d, y)
        }
    }

    fn words_label(text: &str) -> String {
        let n = store::word_count(text);
        format!("{n} {}", if n == 1 { "word" } else { "words" })
    }

    /// Show the open note's last edit and word count under its title.
    fn refresh_note_meta(&self) {
        let notes = self.notes.borrow();
        let meta = self.open_note.get().and_then(|id| notes.get(id)).map(|n| {
            let (_, _, _, _, hr, mi) = win::local_time(n.updated_ms);
            let time = format!("{}:{:02} {}", if hr % 12 == 0 { 12 } else { hr % 12 }, mi, if hr < 12 { "AM" } else { "PM" });
            let day = Self::day_label(n.updated_ms);
            let day = if matches!(day.as_str(), "Today" | "Yesterday") { day.to_lowercase() } else { day };
            format!("Edited {day} at {time} · {}", Self::words_label(&n.text))
        });
        self.state().set_note_meta(meta.unwrap_or_default().into());
    }

    /// The list (filtered by the search) and the open note's details.
    fn refresh_notes(&self) {
        let query = self.note_query.borrow().trim().to_lowercase();
        let notes = self.notes.borrow();
        let listed: Vec<NoteEntry> = notes
            .items
            .iter()
            .filter(|n| query.is_empty() || n.text.to_lowercase().contains(&query))
            .map(|n| {
                let title = store::note_title(&n.text);
                NoteEntry {
                    id: n.id.to_string().into(),
                    title: if title.is_empty() { "Untitled note".into() } else { title.into() },
                    meta: format!("{} · {}", Self::day_label(n.updated_ms), Self::words_label(&n.text)).into(),
                }
            })
            .collect();
        let s = self.state();
        s.set_notes_total(notes.items.len() as i32);
        s.set_notes(ModelRc::new(VecModel::from(listed)));
        drop(notes);
        self.refresh_note_meta();
    }

    /// Put a note (or nothing) into the editor.
    fn show_note(&self, id: Option<u64>) {
        let notes = self.notes.borrow();
        let note = id.and_then(|id| notes.get(id));
        let (title, body) = note.map(|n| store::split_note(&n.text)).unwrap_or_default();
        self.open_note.set(note.map(|n| n.id));
        let s = self.state();
        s.set_note_id(note.map(|n| n.id.to_string()).unwrap_or_default().into());
        s.set_note_title(title.into());
        s.set_note_body(body.into());
        drop(notes);
        self.refresh_note_meta();
    }

    /// Write pending edits to disk.
    fn flush_notes(&self) {
        self.save_timer.stop();
        self.notes.borrow_mut().flush();
    }

    /// Notes left blank (a "New note" that was never typed into) go away.
    fn prune_blank_notes(&self) {
        let blank: Vec<u64> = self.notes.borrow().items.iter().filter(|n| n.text.trim().is_empty()).map(|n| n.id).collect();
        for id in blank {
            self.notes.borrow_mut().remove(id);
            if self.open_note.get() == Some(id) {
                self.show_note(None);
            }
        }
    }

    fn notes_opened(&self) {
        *self.note_query.borrow_mut() = String::new();
        self.prune_blank_notes();
        let open = self.open_note.get().filter(|id| self.notes.borrow().get(*id).is_some());
        let first = self.notes.borrow().items.first().map(|n| n.id);
        self.show_note(open.or(first));
        self.refresh_notes();
    }

    fn open_note_by_id(&self, id: &str) {
        self.flush_notes();
        let id = id.parse::<u64>().ok();
        let leaving = self.open_note.get();
        self.show_note(id);
        if leaving != self.open_note.get() {
            self.prune_blank_notes();
        }
        self.refresh_notes();
    }

    fn new_note(&self) {
        *self.note_query.borrow_mut() = String::new();
        let reuse = self.open_note.get().filter(|id| self.notes.borrow().get(*id).is_some_and(|n| n.text.trim().is_empty()));
        let id = reuse.unwrap_or_else(|| self.notes.borrow_mut().add(store::now_ms()));
        self.show_note(Some(id));
        self.refresh_notes();
    }

    /// The editor changed (typed by hand): keep the text in memory and save shortly after the typing stops.
    fn edit_note(self: &Rc<Self>, title: &str, body: &str) {
        let Some(id) = self.open_note.get() else { return };
        let text = store::join_note(title, body);
        if self.notes.borrow_mut().set_text(id, &text, store::now_ms()) {
            self.refresh_notes();
            let weak = Rc::downgrade(self);
            self.save_timer.start(slint::TimerMode::SingleShot, std::time::Duration::from_millis(600), move || {
                if let Some(a) = weak.upgrade() {
                    a.flush_notes();
                }
            });
        }
    }

    fn delete_note(&self) {
        let Some(id) = self.open_note.get() else { return };
        self.flush_notes();
        self.notes.borrow_mut().remove(id);
        let next = self.notes.borrow().items.first().map(|n| n.id);
        self.show_note(next);
        self.refresh_notes();
    }

    fn copy_note(&self) {
        let Some(id) = self.open_note.get() else { return };
        let text = self.notes.borrow().get(id).map(|n| n.text.clone()).unwrap_or_default();
        if let Ok(mut c) = arboard::Clipboard::new() {
            let _ = c.set_text(text);
        }
    }

    /// A dictation just finished. If you are looking at an open note in Hamsa's own window, it goes into that note
    /// (returns true) instead of being pasted. Runs on the UI thread.
    fn deliver_to_note(&self, text: &str) -> bool {
        let Some(id) = self.open_note.get() else { return false };
        let foreground = win::hwnd(self.main.window()).is_some_and(win::is_foreground);
        if self.state().get_page() != 4 || !foreground {
            return false;
        }
        let current = self.notes.borrow().get(id).map(|n| n.text.clone());
        let Some(current) = current else { return false };
        let text = store::append_text(&current, text);
        self.notes.borrow_mut().set_text(id, &text, store::now_ms());
        self.flush_notes();
        let (title, body) = store::split_note(&text);
        let s = self.state();
        s.set_note_title(title.into());
        s.set_note_body(body.into());
        self.refresh_notes();
        true
    }

    // ---------- Refreshing what the windows show ----------

    fn apply_theme(&self) {
        let choice = self.settings.borrow().theme.clone();
        let dark = match choice.as_str() {
            "light" => false,
            "dark" => true,
            _ => win::system_dark(),
        };
        self.main.global::<Theme>().set_dark(dark);
        self.bar.win.global::<Theme>().set_dark(dark);
        self.setup.win.global::<Theme>().set_dark(dark);
        self.state().set_theme_choice(match choice.as_str() {
            "light" => 1,
            "dark" => 2,
            _ => 0,
        });
    }

    fn refresh_shortcut(&self) {
        let l = labels(&self.prefs.get().shortcut);
        self.state().set_shortcut(strings(l.clone()));
        self.setup.state().set_shortcut(strings(l.clone()));
        self.bar.set_shortcut(&l);
    }

    fn refresh_microphones(&self) {
        let (devices, default) = hamsa_core::audio::list_inputs();
        let mut options = vec![match default {
            Some(d) => format!("Windows default ({d})"),
            None => "Windows default".to_string(),
        }];
        options.extend(devices.iter().cloned());
        let saved = self.settings.borrow().microphone.clone();
        let index = devices.iter().position(|d| *d == saved).map_or(0, |i| i + 1);
        if let Ok(mut m) = self.dictation.microphone.lock() {
            *m = if index > 0 { Some(saved) } else { None };
        }
        self.setup.state().set_microphones(strings(options.clone()));
        self.setup.state().set_microphone(index as i32);
        self.state().set_microphones(strings(options));
        self.state().set_microphone(index as i32);
        *self.microphones.borrow_mut() = devices;
    }

    fn refresh_vocabulary(&self) {
        let v = self.prefs.get().vocabulary;
        let (query, filter) = self.dict_view.borrow().clone();
        let entry = |index: usize, r: &Rule, x: f32, y: f32| DictEntry {
            spoken: r.spoken.as_str().into(),
            text: r.text.as_str().into(),
            same: r.is_word(),
            index: index as i32,
            x,
            y,
        };

        // Dictionary: plain words become chips, the rest rows. Counts follow the search; the filter picks what shows.
        let found: Vec<(usize, &Rule)> = v.dictionary.iter().enumerate().filter(|(_, r)| r.matches(&query)).collect();
        let words: Vec<(usize, &Rule)> = found.iter().copied().filter(|(_, r)| r.is_word()).collect();
        let replacements: Vec<(usize, &Rule)> = found.iter().copied().filter(|(_, r)| !r.is_word()).collect();
        let shown_words = if filter == 2 { Vec::new() } else { words.clone() };
        let shown_repl = if filter == 1 { Vec::new() } else { replacements.clone() };

        // Before the page reports its width, assume the usual one.
        let avail = if self.chip_avail.get() > 1.0 { self.chip_avail.get() } else { 784.0 };
        let widths: Vec<f32> = shown_words.iter().map(|(_, r)| flow::chip_width(&r.text)).collect();
        let (spots, height) = flow::flow(&widths, avail, flow::CHIP_HEIGHT, flow::GAP);
        let chips: Vec<DictEntry> = shown_words.iter().zip(&spots).map(|((i, r), (x, y))| entry(*i, r, *x, *y)).collect();
        let rows: Vec<DictEntry> = shown_repl.iter().map(|(i, r)| entry(*i, r, 0.0, 0.0)).collect();

        let snippets: Vec<SnippetEntry> = v.snippets.iter().enumerate().map(|(i, r)| snippet_entry(i, r)).collect();
        let snippet_query = self.snippet_query.borrow().clone();
        let listed: Vec<SnippetEntry> = v
            .snippets
            .iter()
            .enumerate()
            .filter(|(_, r)| r.matches(&snippet_query))
            .map(|(i, r)| snippet_entry(i, r))
            .collect();

        let s = self.state();
        s.set_dict_total(v.dictionary.len() as i32);
        s.set_count_all(found.len() as i32);
        s.set_count_words(words.len() as i32);
        s.set_count_repl(replacements.len() as i32);
        s.set_chips(ModelRc::new(VecModel::from(chips)));
        s.set_chips_height(height);
        s.set_replacements(ModelRc::new(VecModel::from(rows)));
        s.set_snippets(ModelRc::new(VecModel::from(snippets)));
        s.set_snippet_list(ModelRc::new(VecModel::from(listed)));
    }

    fn refresh_history(&self) {
        const DAY: i64 = 24 * 60 * 60 * 1000;
        const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
        const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        let h = self.history.borrow();
        let now = store::now_ms();
        let date = |ms: i64| {
            let (y, m, d, _, _, _) = win::local_time(ms);
            (y, m, d)
        };
        let (today, yesterday) = (date(now), date(now - DAY));

        let mut groups: Vec<(String, Vec<HistoryItem>)> = Vec::new();
        for item in h.items.iter().take(300) {
            let (y, mo, d, wd, hr, mi) = win::local_time(item.at);
            let label = if (y, mo, d) == today {
                "TODAY".to_string()
            } else if (y, mo, d) == yesterday {
                "YESTERDAY".to_string()
            } else {
                format!("{}, {} {}", DAYS[wd as usize % 7], MONTHS[(mo as usize + 11) % 12], d).to_uppercase()
            };
            let time = format!("{}:{:02} {}", if hr % 12 == 0 { 12 } else { hr % 12 }, mi, if hr < 12 { "AM" } else { "PM" });
            let row = HistoryItem { time: time.into(), text: item.text.as_str().into() };
            match groups.last_mut() {
                Some((l, rows)) if *l == label => rows.push(row),
                _ => groups.push((label, vec![row])),
            }
        }
        let model: Vec<HistoryGroup> = groups
            .into_iter()
            .map(|(label, items)| HistoryGroup { label: label.into(), items: ModelRc::new(VecModel::from(items)) })
            .collect();
        let s = self.state();
        s.set_history(ModelRc::new(VecModel::from(model)));

        // Stats: streak, words this week, words per minute.
        let mut streak = 0;
        let mut day = now;
        while h.items.iter().any(|i| date(i.at) == date(day)) {
            streak += 1;
            day -= DAY;
        }
        let week: u32 = h.items.iter().filter(|i| i.at > now - 7 * DAY).map(|i| i.words).sum();
        let words: u64 = h.items.iter().map(|i| i.words as u64).sum();
        let ms: u64 = h.items.iter().map(|i| i.duration_ms).sum();
        let wpm = if ms > 0 { (words as f64 / (ms as f64 / 60000.0)).round() as u64 } else { 0 };
        s.set_streak(format!("{streak} day streak").into());
        s.set_week(format!("{week} words this week").into());
        s.set_wpm(format!("{wpm} wpm").into());
    }

    fn refresh_bar_settings(&self) {
        let l = self.prefs.get().bar;
        let s = self.state();
        s.set_bar_visible(l.visible);
        self.setup.state().set_bar_visible(l.visible);
        s.set_bar_preview(l.preview);
        s.set_bar_position(match l.placement {
            hamsa_core::Placement::Bottom => 0,
            hamsa_core::Placement::Top => 1,
            hamsa_core::Placement::Custom { .. } => 2,
        });
    }

    fn init_languages(&self) {
        let mut options = vec!["Auto-detect".to_string()];
        options.extend(LANGUAGES.iter().map(|(_, n)| n.to_string()));
        let code = self.prefs.get().language;
        let index = LANGUAGES.iter().position(|(c, _)| *c == code).map_or(0, |i| i + 1);
        self.state().set_languages(strings(options));
        self.state().set_language(index as i32);
        // Shown as Auto / CPU / GPU (the canvas order).
        self.state().set_compute(match self.models.compute() {
            phonon::Compute::Auto => 0,
            phonon::Compute::Cpu => 1,
            phonon::Compute::Gpu => 2,
        });
        self.state().set_models_dir(self.models.dir().display().to_string().into());
    }

    /// Keep beta.json in step with the Darija choice (Phonon choices stay in the shared files, as before).
    fn save_speech_override(&self) {
        let now = self.models.speech_override().map(|m| m.id().to_string());
        let mut s = self.settings.borrow_mut();
        if s.speech_override != now {
            s.speech_override = now;
            store::save_settings(&self.data_dir, &s);
        }
    }

    fn refresh_models(&self, list: &[hamsa_core::ModelStatus]) {
        let cards: Vec<ModelCard> = list
            .iter()
            .map(|m| {
                let (tag, tag_kind, blurb, languages) = about(m.id);
                let downloading = m.progress.is_some();
                let status = if m.active {
                    "In use".to_string()
                } else if m.loading {
                    "Loading…".to_string()
                } else if let Some(p) = m.progress {
                    format!("Downloading {}%", (p * 100.0).round())
                } else if m.installed && m.removable && m.device.is_some() {
                    format!("Ready on {}", m.device.clone().unwrap_or_default())
                } else if m.installed {
                    "Downloaded".to_string()
                } else {
                    "Not downloaded".to_string()
                };
                ModelCard {
                    id: m.id.into(),
                    name: m.name.into(),
                    tag: tag.into(),
                    tag_kind,
                    blurb: blurb.into(),
                    status_ok: m.installed && m.progress.is_none(),
                    languages: languages.into(),
                    size: size_label(m.bytes).into(),
                    removable: m.removable,
                    status: status.into(),
                    device: m.device.clone().unwrap_or_default().into(),
                    installed: m.installed,
                    active: m.active,
                    loading: m.loading,
                    downloading,
                    progress: m.progress.unwrap_or(0.0),
                    error: m.error.clone().unwrap_or_default().into(),
                }
            })
            .collect();
        let s = self.state();
        let (darija, phonon): (Vec<ModelCard>, Vec<ModelCard>) = cards.into_iter().partition(|c| c.removable);
        s.set_models(ModelRc::new(VecModel::from(phonon)));
        s.set_darija_models(ModelRc::new(VecModel::from(darija)));

        let active = list.iter().find(|m| m.active);
        let loading = list.iter().find(|m| m.loading);
        let downloading = list.iter().find(|m| m.progress.is_some());
        s.set_model_ok(active.is_some());
        s.set_model_title(active.or(loading).or(downloading).map_or("No speech model", |m| m.name).into());
        s.set_model_line(
            match (active, loading, downloading) {
                (Some(m), _, _) if m.device.as_deref().is_some_and(|d| d.starts_with("GPU")) => {
                    "Running on your graphics card. Ready to dictate.".to_string()
                }
                (Some(m), _, _) if m.removable => match m.device.as_deref() {
                    Some(d) => format!("Moroccan Darija, running on the {d}. Ready to dictate."),
                    None => "Moroccan Darija. Loads when you dictate.".to_string(),
                },
                (Some(_), _, _) => "Running on this device. Ready to dictate.".to_string(),
                (None, Some(_), _) => "Loading…".to_string(),
                (None, None, Some(m)) => format!("Downloading {}%", (m.progress.unwrap_or(0.0) * 100.0).round()),
                _ => "Download a model to start dictating.".to_string(),
            }
            .into(),
        );
        s.set_has_model(active.is_some());
        s.set_active_name(active.map_or("", |m| m.name).into());
        s.set_multilingual_active(active.is_some_and(|m| m.id == MULTILINGUAL));
        s.set_darija_active(active.is_some_and(|m| m.removable));
        let busy = active.or(loading);
        s.set_device_line(match busy {
            Some(m) if m.loading => "…".into(),
            Some(m) => m.device.clone().unwrap_or_default().into(),
            None => "".into(),
        });
        let note = busy
            .filter(|m| m.device.as_deref() == Some("CPU") && self.models.compute() != phonon::Compute::Cpu)
            .and_then(|m| m.device_note.clone())
            .map(|n| format!("Couldn't use a graphics card: {n}."))
            .unwrap_or_default();
        s.set_device_note(note.into());
        self.refresh_setup(list);
    }

    // ---------- What the windows ask for ----------

    fn connect(self: &Rc<Self>) {
        let s = self.state();
        let weak = Rc::downgrade(self);
        let with = move |f: &dyn Fn(&Rc<App>)| {
            if let Some(a) = weak.upgrade() {
                f(&a)
            }
        };

        let w = with.clone();
        s.on_try_dictation(move || w(&|a| {
            let _ = a.dictation.tx.send(Control::StartManual);
        }));
        s.on_copy_text(|t| {
            if let Ok(mut c) = arboard::Clipboard::new() {
                let _ = c.set_text(t.to_string());
            }
        });

        let w = with.clone();
        s.on_add_word(move |word, heard| {
            let ok = std::cell::Cell::new(false);
            w(&|a| ok.set(a.add_word(word.as_str(), heard.as_str())));
            ok.get()
        });
        let w = with.clone();
        s.on_update_word(move |i, word, heard| {
            let ok = std::cell::Cell::new(false);
            w(&|a| ok.set(a.update_word(i, word.as_str(), heard.as_str())));
            ok.get()
        });
        let w = with.clone();
        s.on_dict_view(move |query, filter| w(&|a| {
            *a.dict_view.borrow_mut() = (query.to_string(), filter);
            a.refresh_vocabulary();
        }));
        let w = with.clone();
        s.on_chip_width(move |width| w(&|a| {
            if (a.chip_avail.get() - width).abs() > 0.5 {
                a.chip_avail.set(width);
                a.refresh_vocabulary();
            }
        }));
        let w = with.clone();
        s.on_snippet_view(move |query| w(&|a| {
            *a.snippet_query.borrow_mut() = query.to_string();
            a.refresh_vocabulary();
        }));
        let w = with.clone();
        s.on_remove_word(move |i| w(&|a| {
            let mut list = a.prefs.get().vocabulary.dictionary;
            if (i as usize) < list.len() {
                list.remove(i as usize);
                a.save_rules(true, list);
            }
        }));
        let w = with.clone();
        s.on_save_snippet(move |cue, text, edit| {
            let ok = std::cell::Cell::new(false);
            w(&|a| {
                let mut list = a.prefs.get().vocabulary.snippets;
                let rule = Rule { spoken: cue.to_string(), text: text.to_string() };
                if edit >= 0 && (edit as usize) < list.len() {
                    list[edit as usize] = rule;
                } else {
                    list.insert(0, rule);
                }
                ok.set(a.save_rules(false, list));
            });
            ok.get()
        });
        let w = with.clone();
        s.on_remove_snippet(move |i| w(&|a| {
            let mut list = a.prefs.get().vocabulary.snippets;
            if (i as usize) < list.len() {
                list.remove(i as usize);
                a.save_rules(false, list);
            }
        }));

        let w = with.clone();
        s.on_notes_opened(move || w(&|a| a.notes_opened()));
        let w = with.clone();
        s.on_notes_view(move |query| w(&|a| {
            *a.note_query.borrow_mut() = query.to_string();
            a.refresh_notes();
        }));
        let w = with.clone();
        s.on_new_note(move || w(&|a| a.new_note()));
        let w = with.clone();
        s.on_open_note(move |id| w(&|a| a.open_note_by_id(id.as_str())));
        let w = with.clone();
        s.on_edit_note(move |title, body| w(&|a| a.edit_note(title.as_str(), body.as_str())));
        let w = with.clone();
        s.on_copy_note(move || w(&|a| a.copy_note()));
        let w = with.clone();
        s.on_delete_note(move || w(&|a| a.delete_note()));
        let w = with.clone();
        s.on_page_changed(move |_| w(&|a| {
            a.flush_notes();
            if a.state().get_page() != 4 {
                a.prune_blank_notes();
                a.refresh_notes();
            }
        }));

        let w = with.clone();
        s.on_set_theme(move |i| w(&|a| {
            a.settings.borrow_mut().theme = ["system", "light", "dark"][i.clamp(0, 2) as usize].to_string();
            store::save_settings(&a.data_dir, &a.settings.borrow());
            a.apply_theme();
        }));

        let w = with.clone();
        s.on_begin_capture(move || w(&|a| {
            a.state().set_capture_mode(1);
            hamsa_core::hotkey::begin_capture();
        }));
        let w = with.clone();
        s.on_cancel_capture(move || w(&|a| {
            hamsa_core::hotkey::cancel_capture();
            a.state().set_capture_mode(0);
        }));
        let w = with.clone();
        s.on_save_capture(move || w(&|a| {
            let keys = a.captured.borrow().clone();
            if hamsa_core::hotkey::validate(&keys).is_ok() {
                hamsa_core::hotkey::set_combo(&keys);
                a.prefs.update(|p| p.shortcut = keys);
                a.refresh_shortcut();
                a.state().set_capture_mode(0);
            }
        }));

        let w = with.clone();
        s.on_set_formatting(move |capitalize, end, fillers, smart, lists, bullet| w(&|a| {
            let f = hamsa_core::Formatting {
                capitalize,
                end_punctuation: end,
                remove_fillers: fillers,
                smart_punctuation: smart,
                lists,
                bullet: if bullet == 1 { hamsa_core::Bullet::Dash } else { hamsa_core::Bullet::Dot },
            };
            if let Ok(mut cur) = a.dictation.formatting.lock() {
                *cur = f;
            }
            store::save_formatting(&a.data_dir, &f);
        }));

        let w = with.clone();
        s.on_set_hands_free(move |on| w(&|a| {
            a.settings.borrow_mut().hands_free = on;
            store::save_settings(&a.data_dir, &a.settings.borrow());
            a.dictation.hands_free.store(on, std::sync::atomic::Ordering::SeqCst);
            a.bar.win.set_hands_free(on);
        }));

        let w = with.clone();
        s.on_set_start_with_windows(move |on| w(&|a| {
            crate::autostart::set_enabled(on);
            // Show what the registry really says, in case the write failed.
            a.state().set_start_with_windows(crate::autostart::is_enabled());
        }));
        let w = with.clone();
        s.on_run_setup(move || w(&|a| a.show_setup(1)));
        self.connect_setup();

        let w = with.clone();
        s.on_set_cleanup(move |on| w(&|a| a.choose_cleanup(on)));
        let w = with.clone();
        s.on_set_cleanup_size(move |i| w(&|a| {
            let size = if i == 1 { hamsa_core::CleanupSize::Small } else { hamsa_core::CleanupSize::Standard };
            a.settings.borrow_mut().cleanup_size = size.name().to_string();
            store::save_settings(&a.data_dir, &a.settings.borrow());
            a.dictation.cleanup.set_size(size);
        }));
        let w = with.clone();
        s.on_download_cleanup(move || w(&|a| a.dictation.cleanup.download()));
        let w = with.clone();
        s.on_remove_cleanup(move || w(&|a| {
            // Without the download there is nothing to run, so the switch goes off too.
            a.settings.borrow_mut().smart_cleanup = false;
            store::save_settings(&a.data_dir, &a.settings.borrow());
            a.state().set_cleanup_on(false);
            a.dictation.cleanup.set_enabled(false);
            a.dictation.cleanup.remove();
        }));

        let w = with.clone();
        s.on_set_microphone(move |i| w(&|a| a.choose_microphone(i)));

        let w = with.clone();
        s.on_set_bar(move |visible, preview, position| w(&|a| {
            a.bar.set_layout(visible, preview, position);
            a.refresh_bar_settings();
        }));
        let w = with.clone();
        s.on_set_compute(move |i| w(&|a| {
            if let Some(c) = models::compute_from(["auto", "cpu", "gpu"][i.clamp(0, 2) as usize]) {
                a.models.set_compute(c);
            }
        }));
        let w = with.clone();
        s.on_set_language(move |i| w(&|a| {
            let code = if i <= 0 { "auto".to_string() } else { LANGUAGES[(i as usize - 1).min(24)].0.to_string() };
            a.prefs.update(|p| p.language = code);
        }));
        let w = with.clone();
        s.on_download_model(move |id| w(&|a| a.models.download(id.to_string())));
        let w = with.clone();
        s.on_use_model(move |id| w(&|a| a.use_model(id.as_str())));
        let w = with.clone();
        s.on_remove_model(move |id| w(&|a| {
            a.models.remove(id.as_str());
            a.save_speech_override();
        }));
        let w = with.clone();
        s.on_open_folder(move || w(&|a| {
            let _ = std::process::Command::new("explorer").arg(a.models.dir()).spawn();
        }));

        // The bar
        let b = &self.bar.win;
        let w = with.clone();
        b.on_cancel(move || w(&|a| {
            let _ = a.dictation.tx.send(Control::Cancel);
        }));
        let w = with.clone();
        b.on_stop(move || w(&|a| {
            let _ = a.dictation.tx.send(Control::Stop);
        }));
        let w = with.clone();
        b.on_open_menu(move || w(&|a| a.bar.open_menu()));
        let w = with.clone();
        b.on_start_drag(move || w(&|a| a.bar.start_drag()));
        let w = with.clone();
        b.on_reset_position(move || w(&|a| {
            a.bar.reset_position();
            a.refresh_bar_settings();
        }));
        let w = with.clone();
        b.on_resized(move || w(&|a| a.bar.place()));
        let w = with;
        b.on_menu_action(move |action| w(&|a| a.menu_action(action.as_str())));

        // Closing the main window keeps Hamsa Beta running in the tray.
        self.main.window().on_close_requested(|| slint::CloseRequestResponse::HideWindow);
    }

    fn menu_action(&self, action: &str) {
        let l = self.prefs.get().bar;
        let position = match l.placement {
            hamsa_core::Placement::Bottom => 0,
            hamsa_core::Placement::Top => 1,
            hamsa_core::Placement::Custom { .. } => 2,
        };
        match action {
            "paste" | "copy" => {
                self.bar.close_menu();
                if let Some(text) = hamsa_core::dictation::last_text() {
                    if action == "copy" {
                        if let Ok(mut c) = arboard::Clipboard::new() {
                            let _ = c.set_text(text);
                        }
                    } else {
                        // The bar never takes focus, so the app you were in still has it.
                        std::thread::spawn(move || {
                            std::thread::sleep(std::time::Duration::from_millis(120));
                            let _ = hamsa_core::paste::paste_text(&text);
                        });
                    }
                }
            }
            "preview" => self.bar.set_layout(l.visible, !l.preview, position),
            "idle" => self.bar.set_layout(!l.visible, l.preview, position),
            "top" => self.bar.set_layout(l.visible, l.preview, 1),
            "bottom" => self.bar.set_layout(l.visible, l.preview, 0),
            "open" => {
                self.bar.close_menu();
                self.show_main();
            }
            "settings" => {
                self.bar.close_menu();
                self.state().set_settings_tab(0);
                self.state().set_page(5);
                self.show_main();
            }
            "quit" => {
                let _ = slint::quit_event_loop();
            }
            _ => {}
        }
        if matches!(action, "preview" | "idle" | "top" | "bottom") {
            self.bar.close_menu();
            self.refresh_bar_settings();
        }
    }

    fn add_word(&self, word: &str, heard: &str) -> bool {
        let text = word.trim();
        if text.is_empty() {
            return false;
        }
        let spoken = if heard.trim().is_empty() { text } else { heard.trim() };
        let mut list = self.prefs.get().vocabulary.dictionary;
        // Re-adding an entry replaces it.
        list.retain(|r| r.spoken.to_lowercase() != spoken.to_lowercase());
        list.insert(0, Rule { spoken: spoken.to_string(), text: text.to_string() });
        self.save_rules(true, list)
    }

    /// Change one dictionary entry in place (the add row's "Save" while editing).
    fn update_word(&self, index: i32, word: &str, heard: &str) -> bool {
        let text = word.trim();
        let mut list = self.prefs.get().vocabulary.dictionary;
        if text.is_empty() || index < 0 || index as usize >= list.len() {
            return false;
        }
        let spoken = if heard.trim().is_empty() { text } else { heard.trim() };
        list[index as usize] = Rule { spoken: spoken.to_string(), text: text.to_string() };
        self.save_rules(true, list)
    }

    /// Validate and save the dictionary (or snippets). Shows the error and returns false if it can't be saved.
    fn save_rules(&self, dictionary: bool, rules: Vec<Rule>) -> bool {
        let rules: Vec<Rule> = rules
            .into_iter()
            .map(|r| Rule { spoken: r.spoken.split_whitespace().collect::<Vec<_>>().join(" "), text: r.text.trim_end().to_string() })
            .collect();
        let check = hamsa_core::vocabulary::validate(&rules, if dictionary { "entry" } else { "snippet" });
        let s = self.state();
        match check {
            Err(e) => {
                if dictionary {
                    s.set_words_error(e.into());
                } else {
                    s.set_snippets_error(e.into());
                }
                false
            }
            Ok(()) => {
                self.prefs.update(|p| {
                    if dictionary {
                        p.vocabulary.dictionary = rules;
                    } else {
                        p.vocabulary.snippets = rules;
                    }
                });
                s.set_words_error("".into());
                s.set_snippets_error("".into());
                self.refresh_vocabulary();
                true
            }
        }
    }

    /// The microphone picker (Settings and the setup window share it). `i` 0 is the Windows default.
    fn choose_microphone(self: &Rc<Self>, i: i32) {
        let name = if i > 0 { self.microphones.borrow().get(i as usize - 1).cloned() } else { None };
        if let Ok(mut m) = self.dictation.microphone.lock() {
            *m = name.clone();
        }
        self.settings.borrow_mut().microphone = name.unwrap_or_default();
        store::save_settings(&self.data_dir, &self.settings.borrow());
        self.state().set_microphone(i);
        self.setup.state().set_microphone(i);
        self.setup_microphone_changed();
    }

    /// Smart cleanup's switch (Settings and the setup window share it): turning it on starts the download if needed.
    fn choose_cleanup(&self, on: bool) {
        self.settings.borrow_mut().smart_cleanup = on;
        store::save_settings(&self.data_dir, &self.settings.borrow());
        self.dictation.cleanup.set_enabled(on);
        self.state().set_cleanup_on(on);
        self.setup.state().set_cleanup_on(on);
    }

    /// Make `id` the model that transcribes, and keep the Darija choice in beta.json.
    fn use_model(&self, id: &str) {
        self.models.activate(id.to_string());
        self.save_speech_override();
    }

    pub fn show_main(&self) {
        // While the setup window is open it is the one to bring forward (the main window stays hidden).
        if self.setup.is_open() {
            self.raise_setup();
            return;
        }
        let _ = self.main.show();
        if let Some(h) = win::hwnd(self.main.window()) {
            win::bring_to_front(h);
        }
    }

    /// Developer aid for checking screens without clicking: HAMSA_BETA_PAGE=0..6, HAMSA_BETA_TAB=0..3,
    /// HAMSA_BETA_THEME=light|dark, HAMSA_BETA_DEMO=listening|locked|preview|transcribing|menu|hint|cycle shows a bar state with sample text.
    pub fn apply_dev_options(self: &Rc<Self>) {
        if let Some(p) = std::env::var("HAMSA_BETA_PAGE").ok().and_then(|v| v.parse().ok()) {
            self.state().set_page(p);
        }
        if let Ok(t) = std::env::var("HAMSA_BETA_THEME") {
            let dark = t == "dark";
            self.main.global::<Theme>().set_dark(dark);
            self.bar.win.global::<Theme>().set_dark(dark);
            self.setup.win.global::<Theme>().set_dark(dark);
        }
        if let Some(t) = std::env::var("HAMSA_BETA_TAB").ok().and_then(|v| v.parse().ok()) {
            self.state().set_settings_tab(t);
        }
        // HAMSA_BETA_BAR=top,hide-idle: bar at the top of the screen and/or hidden while idle (for checking the motion).
        if let Ok(opts) = std::env::var("HAMSA_BETA_BAR") {
            self.bar.set_layout(!opts.contains("hide-idle"), true, if opts.contains("top") { 1 } else { 0 });
        }
        if let Ok(demo) = std::env::var("HAMSA_BETA_DEMO") {
            let b = &self.bar;
            match demo.as_str() {
                "listening" => b.set_phase(BarPhase::Listening, 0.7),
                "locked" => {
                    b.set_phase(BarPhase::Listening, 0.7);
                    b.set_locked(true);
                }
                "hint" => b.win.set_force_hint(true),
                "preview" => {
                    b.set_phase(BarPhase::Listening, 0.6);
                    b.set_preview(
                        "so the plan for next week is to ship the beta on Monday, collect feedback until Wednesday and then fix the top three issues",
                        "before Friday",
                    );
                }
                "transcribing" => b.set_phase(BarPhase::Transcribing, 0.0),
                "menu" => b.open_menu(),
                "cycle" => self.demo_cycle(),
                _ => {}
            }
        }
    }


    /// HAMSA_BETA_DEMO=cycle: idle, hover hint, listening, live preview, transcribing, repeat (250 ms per tick).
    fn demo_cycle(&self) {
        const WORDS: &str = "so the plan for next week is to ship the beta on Monday, collect feedback until Wednesday and then fix the top three issues together";
        let bar = self.bar.clone();
        let tick = std::cell::Cell::new(0usize);
        let menu_open = std::cell::Cell::new(false);
        let timer: &'static slint::Timer = Box::leak(Box::new(slint::Timer::default()));
        timer.start(slint::TimerMode::Repeated, std::time::Duration::from_millis(250), move || {
            let n = tick.get();
            tick.set(n + 1);
            let words: Vec<&str> = WORDS.split(' ').collect();
            let (idle, hint, listen, preview, menu) = (6, 6, 6, words.len() + 2, 8);
            let wave = 0.55 + 0.4 * (n as f32 * 0.9).sin();
            let step = n % (idle + hint + listen + preview + 6 + menu);
            bar.win.set_force_hint(false);
            if step < idle + hint + listen + preview + 6 && menu_open.replace(false) {
                bar.close_menu();
            }
            if step < idle {
                bar.set_phase(BarPhase::Idle, 0.0);
            } else if step < idle + hint {
                bar.set_phase(BarPhase::Idle, 0.0);
                bar.win.set_force_hint(true);
            } else if step < idle + hint + listen {
                bar.set_phase(BarPhase::Listening, wave);
            } else if step < idle + hint + listen + preview {
                let k = (step - idle - hint - listen + 1).min(words.len());
                let split = k.saturating_sub(2);
                bar.set_phase(BarPhase::Listening, wave);
                bar.set_preview(&words[..split].join(" "), &words[split..k].join(" "));
            } else if step < idle + hint + listen + preview + 6 {
                bar.set_phase(BarPhase::Transcribing, 0.0);
            } else {
                // The right-click menu opens and closes at the end of the loop.
                bar.set_phase(BarPhase::Idle, 0.0);
                if !menu_open.replace(true) {
                    bar.open_menu();
                }
            }
        });
    }

    /// Save anything typed into a note but not yet written (called as the app quits).
    pub fn save_all(&self) {
        self.flush_notes();
        // Stop the cleanup helper process (a job object also kills it if Hamsa dies without getting here).
        self.dictation.cleanup.shutdown();
    }

    pub fn test_microphone(&self) {
        let _ = self.dictation.tx.send(Control::StartManual);
    }
}

thread_local! {
    /// The app, for core events delivered to the UI thread.
    pub static APP: RefCell<Option<Rc<App>>> = const { RefCell::new(None) };
}
