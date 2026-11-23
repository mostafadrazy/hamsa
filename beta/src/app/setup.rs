//! The first-run setup window: five steps (welcome, speech model, microphone, try it, ready) in their own window.
//!
//! Nothing here is new machinery. The model cards start the same `Models::download` and `activate` the Settings cards
//! use, the microphone picker is Settings' picker, the level meter and "Test" use the core's microphone code and the
//! "Test microphone" recording, smart cleanup's switch is Settings' switch, and "Start Hamsa when Windows starts" goes
//! through `autostart` when "Start dictating" is pressed. Whether it opens at launch is `decide`.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::time::Duration;

use hamsa_core::{audio, dictation::Control, CleanupStatus, Finished, LlmState, ModelStatus};
use slint::{ComponentHandle, Model, ModelRc, Timer, TimerMode, VecModel};

use super::{size_label, App};
use crate::{autostart, store, win, SetupModel, SetupState, SetupWindow};

/// How long the "Test" button records.
const TEST_LENGTH: Duration = Duration::from_secs(3);
/// If the result still hasn't arrived this long after the recording stopped, give up on the test.
const TEST_GUARD: Duration = Duration::from_secs(25);
/// "Too quiet" when the loudest sample of the test stayed under this (the home page already calls < 0.02 "almost
/// nothing"; clear speech peaks well above this)...
const QUIET_PEAK: f32 = 0.05;
/// ...or when the live meter (0 to 1 on the −55 to −10 dB scale) never got past this. "About halfway" is 0.5.
const QUIET_LEVEL: f32 = 0.3;

/// What the app does about setup when it starts.
#[derive(Debug, PartialEq, Eq)]
pub enum Launch {
    /// Open the setup window (the main window stays hidden).
    Setup,
    /// An existing user who already has a model: record that setup is done, show nothing.
    MarkDone,
    /// Nothing to do.
    Nothing,
}

/// Setup opens on its own only for a foreground launch that has neither finished setup before nor any speech model
/// (no Phonon model downloaded and no Darija model). A `--background` launch never shows it.
pub fn decide(setup_done: bool, has_model: bool, background: bool) -> Launch {
    if background || setup_done {
        Launch::Nothing
    } else if has_model {
        Launch::MarkDone
    } else {
        Launch::Setup
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Good,
    Quiet,
}

/// The microphone test's verdict from the loudest sample (0 to 1) and the highest live meter value (0 to 1).
pub fn verdict(peak: f32, max_level: f32) -> Verdict {
    if peak < QUIET_PEAK || max_level < QUIET_LEVEL {
        Verdict::Quiet
    } else {
        Verdict::Good
    }
}

/// Dictated `addition` after the practice box's `current` text, one space between them.
pub fn append_practice(current: &str, addition: &str) -> String {
    let addition = addition.trim();
    if addition.is_empty() {
        return current.to_string();
    }
    if current.is_empty() || current.ends_with(char::is_whitespace) {
        format!("{current}{addition}")
    } else {
        format!("{current} {addition}")
    }
}

/// Card text for each model: name, recommended, languages, speed, accuracy.
fn about(id: &str) -> (&'static str, bool, &'static str, &'static str, &'static str) {
    match id {
        "phonon-2.5-en" => ("Phonon-2.5 English", true, "English only", "Fastest", "Very high for English"),
        "phonon-2.5" => ("Phonon-2.5", false, "Multilingual, detects the language you speak", "Fast", "High across languages"),
        "fihria-asr-0.6b" => ("Darija · fast", false, "Moroccan Darija · fihrIA-ASR 0.6B", "Fast", "Good"),
        "fihria-asr-1.7b" => ("Darija · accurate", false, "Moroccan Darija · fihrIA-ASR 1.7B", "Slower", "Highest"),
        _ => ("Speech model", false, "", "", ""),
    }
}

fn percent(p: f32) -> i32 {
    (p * 100.0).round() as i32
}

/// What the setup window keeps while it is open.
pub struct Setup {
    pub win: SetupWindow,
    open: Cell<bool>,
    /// The model to switch to once its download finishes.
    pending: RefCell<Option<String>>,
    /// The live level meter's microphone, open only while step 3 is showing.
    meter: Rc<RefCell<Option<audio::Recording>>>,
    smoothed: Rc<Cell<f32>>,
    meter_timer: Timer,
    place_timer: Timer,
    /// A "Test" recording is running or its result is awaited.
    testing: Cell<bool>,
    /// Highest live level seen during the test.
    max_level: Cell<f32>,
    stop_timer: Timer,
    guard_timer: Timer,
}

impl Setup {
    pub fn new(win: SetupWindow) -> Self {
        Self {
            win,
            open: Cell::new(false),
            pending: RefCell::new(None),
            meter: Rc::new(RefCell::new(None)),
            smoothed: Rc::new(Cell::new(0.0)),
            meter_timer: Timer::default(),
            place_timer: Timer::default(),
            testing: Cell::new(false),
            max_level: Cell::new(0.0),
            stop_timer: Timer::default(),
            guard_timer: Timer::default(),
        }
    }

    pub fn state(&self) -> SetupState<'_> {
        self.win.global::<SetupState>()
    }

    pub fn is_open(&self) -> bool {
        self.open.get()
    }
}

fn with(weak: &Weak<App>, f: impl FnOnce(&Rc<App>)) {
    if let Some(a) = weak.upgrade() {
        f(&a);
    }
}

impl App {
    // ---------- When it shows ----------

    /// Called at launch: opens setup, or quietly records that an existing user is done. Returns whether it opened.
    pub fn startup_setup(self: &Rc<Self>, background: bool) -> bool {
        let has_model = self.models.status().iter().any(|m| m.installed);
        let done = self.settings.borrow().setup_done;
        match decide(done, has_model, background) {
            Launch::Setup => {
                self.show_setup(1);
                true
            }
            Launch::MarkDone => {
                self.mark_setup_done();
                false
            }
            Launch::Nothing => false,
        }
    }

    fn mark_setup_done(&self) {
        let mut s = self.settings.borrow_mut();
        if !s.setup_done {
            s.setup_done = true;
            store::save_settings(&self.data_dir, &s);
        }
    }

    /// Open the setup window at `step` (1 to 5). The main window is hidden while it is open.
    pub fn show_setup(self: &Rc<Self>, step: i32) {
        let s = &self.setup;
        s.open.set(true);
        self.prepare_setup();
        s.state().set_step(step.clamp(1, 5));
        let _ = self.main.hide();
        let _ = s.win.show();
        // The native window exists once the event loop runs: centre it and bring it forward then.
        let weak = Rc::downgrade(self);
        s.place_timer.start(TimerMode::Repeated, Duration::from_millis(20), move || {
            with(&weak, |a| {
                if let Some(h) = win::hwnd(a.setup.win.window()) {
                    a.setup.place_timer.stop();
                    win::center(h);
                    win::bring_to_front(h);
                }
            });
        });
        self.setup_step_entered(step.clamp(1, 5));
    }

    pub(super) fn raise_setup(&self) {
        let _ = self.setup.win.show();
        if let Some(h) = win::hwnd(self.setup.win.window()) {
            win::bring_to_front(h);
        }
    }

    /// Fresh state each time the window opens.
    fn prepare_setup(&self) {
        let s = &self.setup;
        let st = s.state();
        st.set_practice_text("".into());
        st.set_test_phase(0);
        st.set_verdict(0);
        st.set_heard("".into());
        st.set_test_status("Press Test, then say the sentence.".into());
        st.set_mic_error("".into());
        s.testing.set(false);
        s.pending.borrow_mut().take();
        let first_run = !self.settings.borrow().setup_done;
        // On the first run it defaults to on; running setup again shows what Windows really has.
        st.set_start_windows(first_run || autostart::is_enabled());
        let list = self.models.status();
        let choice = list.iter().find(|m| m.active).or_else(|| list.iter().find(|m| m.installed)).map_or("phonon-2.5-en", |m| m.id);
        st.set_selected_model(choice.into());
        self.refresh_setup_cleanup(&self.dictation.cleanup.status());
        self.refresh_setup_bar();
        self.refresh_setup(&list);
    }

    fn refresh_setup_bar(&self) {
        self.setup.state().set_bar_visible(self.prefs.get().bar.visible);
    }

    /// "Start dictating" (apply the start-with-Windows choice) or the window's X (nothing applied): setup is done.
    pub(super) fn finish_setup(&self, apply_autostart: bool) {
        let s = &self.setup;
        self.setup_meter_stop();
        if s.testing.get() {
            let _ = self.dictation.tx.send(Control::Cancel);
            self.setup_test_reset();
        }
        s.place_timer.stop();
        s.open.set(false);
        self.mark_setup_done();
        if apply_autostart {
            autostart::set_enabled(s.state().get_start_windows());
            self.state().set_start_with_windows(autostart::is_enabled());
        }
        let _ = s.win.hide();
        self.show_main();
    }

    // ---------- Wiring ----------

    pub(super) fn connect_setup(self: &Rc<Self>) {
        let st = self.setup.state();
        let weak = Rc::downgrade(self);

        let w = weak.clone();
        st.on_next(move || with(&w, |a| a.setup_go(a.setup.state().get_step() + 1)));
        let w = weak.clone();
        st.on_back(move || with(&w, |a| a.setup_go(a.setup.state().get_step() - 1)));
        let w = weak.clone();
        st.on_go(move |step| with(&w, |a| a.setup_go(step)));
        let w = weak.clone();
        st.on_finish(move || with(&w, |a| a.finish_setup(true)));
        let w = weak.clone();
        st.on_pick_model(move |id| with(&w, |a| {
            a.setup.state().set_selected_model(id);
            a.refresh_setup_now();
        }));
        let w = weak.clone();
        st.on_download_model(move |id| with(&w, |a| {
            a.setup.state().set_selected_model(id.clone());
            a.setup_download(id.as_str());
        }));
        let w = weak.clone();
        st.on_pick_mic(move |i| with(&w, |a| a.choose_microphone(i)));
        let w = weak.clone();
        st.on_run_test(move || with(&w, |a| a.setup_test()));
        let w = weak.clone();
        st.on_set_cleanup(move |on| with(&w, |a| a.choose_cleanup(on)));
        let w = weak.clone();
        st.on_set_bar(move |visible| with(&w, |a| {
            let l = a.prefs.get().bar;
            let position = match l.placement {
                hamsa_core::Placement::Bottom => 0,
                hamsa_core::Placement::Top => 1,
                hamsa_core::Placement::Custom { .. } => 2,
            };
            a.bar.set_layout(visible, l.preview, position);
            a.refresh_bar_settings();
        }));

        // Closing with the X counts as done, like "Start dictating" (but applies nothing).
        let w = weak;
        self.setup.win.window().on_close_requested(move || {
            with(&w, |a| a.finish_setup(false));
            slint::CloseRequestResponse::HideWindow
        });
    }

    /// Move to a step. Leaving the model step forward starts or applies the choice; the meter runs only on step 3.
    fn setup_go(self: &Rc<Self>, to: i32) {
        let to = to.clamp(1, 5);
        let st = self.setup.state();
        let from = st.get_step();
        if from == 2 && to > 2 {
            self.setup_apply_model();
        }
        if from == 3 && to != 3 && self.setup.testing.get() {
            let _ = self.dictation.tx.send(Control::Cancel);
            self.setup_test_reset();
        }
        st.set_step(to);
        self.setup_step_entered(to);
        self.refresh_setup_now();
    }

    fn setup_step_entered(self: &Rc<Self>, step: i32) {
        if step == 3 {
            self.setup_meter_start();
        } else {
            self.setup_meter_stop();
        }
    }

    // ---------- Step 2: the speech model ----------

    fn refresh_setup_now(&self) {
        self.refresh_setup(&self.models.status());
    }

    /// The model cards, whether a model is ready, the "still downloading" line, the Ready step's summary, and the
    /// switch to the chosen model once its download has finished.
    pub(super) fn refresh_setup(&self, list: &[ModelStatus]) {
        let st = self.setup.state();
        let cards: Vec<SetupModel> = list
            .iter()
            .map(|m| {
                let (name, recommended, langs, speed, acc) = about(m.id);
                SetupModel {
                    id: m.id.into(),
                    name: name.into(),
                    recommended,
                    langs: langs.into(),
                    size: size_label(m.bytes).into(),
                    speed: speed.into(),
                    acc: acc.into(),
                    installed: m.installed,
                    downloading: m.progress.is_some(),
                    progress: m.progress.unwrap_or(0.0),
                    percent: percent(m.progress.unwrap_or(0.0)),
                    status: if m.active {
                        "In use"
                    } else if m.loading {
                        "Loading…"
                    } else {
                        "Downloaded"
                    }
                    .into(),
                    error: m.error.clone().unwrap_or_default().into(),
                    active: m.active,
                }
            })
            .collect();
        let mut cards = cards;
        // Developer aid for checking the progress bars without a real download: HAMSA_BETA_SETUP_DEMO=download.
        if std::env::var("HAMSA_BETA_SETUP_DEMO").as_deref() == Ok("download") {
            for (card, p) in cards.iter_mut().zip([0.42f32, 0.18, 0.09, 0.03]) {
                (card.installed, card.downloading, card.progress, card.percent) = (false, true, p, percent(p));
            }
        }
        st.set_models(ModelRc::new(VecModel::from(cards)));

        let active = list.iter().find(|m| m.active);
        st.set_model_ready(active.is_some());
        let pending = self.setup.pending.borrow().clone();
        let downloading = pending
            .as_deref()
            .and_then(|id| list.iter().find(|m| m.id == id && m.progress.is_some()))
            .or_else(|| list.iter().find(|m| m.progress.is_some()));
        st.set_model_line(
            if let Some(m) = downloading {
                format!("Your model is still downloading ({} %)", percent(m.progress.unwrap_or(0.0)))
            } else if list.iter().any(|m| m.loading) {
                "Your model is loading…".to_string()
            } else {
                "No speech model yet. Choose one in step 2.".to_string()
            }
            .into(),
        );

        self.setup_activate_pending(list);

        // The Ready step's summary of the model.
        let chosen = st.get_selected_model();
        if let Some(m) = list.iter().find(|m| m.id == chosen.as_str()) {
            st.set_summary_model(about(m.id).0.into());
            st.set_summary_model_note(
                match m.progress {
                    Some(p) => format!("· downloading {} %", percent(p)),
                    None if m.installed => format!("· {}", size_label(m.bytes)),
                    None => "· not downloaded".to_string(),
                }
                .into(),
            );
        }
        self.refresh_setup_mic();
    }

    /// Once the chosen model has finished downloading, switch to it through `activate` (a Phonon model loads by itself
    /// when its download ends; a Darija model, or any model while a Darija one is in use, needs the switch).
    fn setup_activate_pending(&self, list: &[ModelStatus]) {
        let Some(id) = self.setup.pending.borrow().clone() else { return };
        let Some(m) = list.iter().find(|m| m.id == id) else { return };
        if m.progress.is_some() || m.loading {
            return;
        }
        if !m.installed {
            if m.error.is_some() {
                self.setup.pending.borrow_mut().take(); // the download failed; the card shows why
            }
            return;
        }
        self.setup.pending.borrow_mut().take();
        if !m.active {
            self.use_model(&id);
        }
    }

    /// Start the chosen model's real download (or note that it is already running), to be activated when it ends.
    fn setup_download(&self, id: &str) {
        let list = self.models.status();
        let Some(m) = list.iter().find(|m| m.id == id) else { return };
        if m.installed {
            return;
        }
        *self.setup.pending.borrow_mut() = Some(id.to_string());
        if m.progress.is_none() {
            self.models.download(id.to_string());
        }
        self.refresh_setup_now();
    }

    /// "Continue" on the model step: download the selected model, or switch to it when it is already installed.
    fn setup_apply_model(&self) {
        let id = self.setup.state().get_selected_model().to_string();
        let list = self.models.status();
        let Some(m) = list.iter().find(|m| m.id == id) else { return };
        if m.installed {
            self.setup.pending.borrow_mut().take();
            if !m.active && !m.loading {
                self.use_model(&id);
            }
        } else {
            self.setup_download(&id);
        }
    }

    // ---------- Step 3: microphone ----------

    fn refresh_setup_mic(&self) {
        let st = self.setup.state();
        let i = st.get_microphone();
        let name = if i >= 0 { st.get_microphones().row_data(i as usize) } else { None };
        st.set_summary_mic(name.unwrap_or_default());
    }

    /// The microphone was changed (anywhere): follow it with the live meter.
    pub(super) fn setup_microphone_changed(self: &Rc<Self>) {
        self.refresh_setup_mic();
        if self.setup.is_open() && self.setup.state().get_step() == 3 && !self.setup.testing.get() {
            self.setup_meter_start();
        }
    }

    /// Open the chosen microphone and show its level while step 3 is visible.
    fn setup_meter_start(self: &Rc<Self>) {
        let s = &self.setup;
        self.setup_meter_stop();
        if s.testing.get() {
            return;
        }
        let device = self.dictation.microphone.lock().ok().and_then(|m| m.clone());
        match audio::start(device) {
            Ok(rec) => {
                *s.meter.borrow_mut() = Some(rec);
                s.state().set_mic_error("".into());
                let (rec, smoothed, win) = (s.meter.clone(), s.smoothed.clone(), s.win.as_weak());
                s.meter_timer.start(TimerMode::Repeated, Duration::from_millis(50), move || {
                    let Some(win) = win.upgrade() else { return };
                    let level = rec.borrow().as_ref().map_or(0.0, |r| audio::meter(r.rms()));
                    // Rises at once, falls a little at a time.
                    smoothed.set(level.max(smoothed.get() * 0.8));
                    win.global::<SetupState>().set_level(smoothed.get());
                });
            }
            Err(e) => s.state().set_mic_error(e.into()),
        }
    }

    /// Close the meter's microphone (leaving step 3, closing the window, or before a test uses the mic).
    fn setup_meter_stop(&self) {
        let s = &self.setup;
        s.meter_timer.stop();
        s.meter.borrow_mut().take();
        s.smoothed.set(0.0);
        s.state().set_level(0.0);
    }

    // ---------- Step 3: the test ----------

    fn setup_test(self: &Rc<Self>) {
        let s = &self.setup;
        if s.testing.get() {
            return;
        }
        self.setup_meter_stop(); // the test recording takes the microphone
        s.testing.set(true);
        s.max_level.set(0.0);
        let st = s.state();
        st.set_test_phase(1);
        st.set_verdict(0);
        st.set_heard("".into());
        st.set_test_status("Say the sentence now".into());
        let _ = self.dictation.tx.send(Control::StartManual);
        let weak = Rc::downgrade(self);
        s.stop_timer.start(TimerMode::SingleShot, TEST_LENGTH, move || with(&weak, |a| a.setup_test_stop()));
    }

    fn setup_test_stop(self: &Rc<Self>) {
        let s = &self.setup;
        if !s.testing.get() {
            return;
        }
        let _ = self.dictation.tx.send(Control::Stop);
        let st = s.state();
        st.set_test_phase(2);
        st.set_test_status("Working it out…".into());
        let weak = Rc::downgrade(self);
        s.guard_timer.start(TimerMode::SingleShot, TEST_GUARD, move || {
            with(&weak, |a| a.setup_test_failed("The test didn't finish. Try again."))
        });
    }

    /// Back to idle after a test, without a result.
    fn setup_test_reset(&self) {
        let s = &self.setup;
        s.testing.set(false);
        s.stop_timer.stop();
        s.guard_timer.stop();
        s.state().set_test_phase(0);
        s.state().set_level(0.0);
    }

    fn setup_test_failed(self: &Rc<Self>, message: &str) {
        self.setup_test_reset();
        let st = self.setup.state();
        st.set_verdict(2);
        st.set_test_status(message.into());
        self.setup_resume_meter();
    }

    fn setup_resume_meter(self: &Rc<Self>) {
        if self.setup.is_open() && self.setup.state().get_step() == 3 {
            self.setup_meter_start();
        }
    }

    /// The live level during a test comes from the dictation's own meter events.
    pub(super) fn setup_level(&self, level: f32) {
        let s = &self.setup;
        if s.testing.get() {
            s.max_level.set(s.max_level.get().max(level));
            s.state().set_level(level);
        }
    }

    pub(super) fn setup_error(&self, message: &str) {
        if !self.setup.testing.get() {
            return;
        }
        // Needs the Rc for the meter restart, so go through the app handle.
        let message = message.to_string();
        super::APP.with(|a| {
            if let Some(app) = a.borrow().as_ref().cloned() {
                app.setup_test_failed(&message);
            }
        });
    }

    /// The test recording's result. Returns true when the setup window took it (so it isn't a normal dictation).
    pub(super) fn setup_finished(&self, f: &Finished) -> bool {
        let s = &self.setup;
        if !s.testing.get() {
            return false;
        }
        let max_level = s.max_level.get();
        self.setup_test_reset();
        let st = s.state();
        st.set_test_phase(3);
        let quiet = verdict(f.peak, max_level) == Verdict::Quiet;
        st.set_verdict(if quiet { 2 } else { 1 });
        st.set_test_status(if quiet { "Too quiet: move closer or pick another mic" } else { "Sounds good" }.into());
        // What Hamsa heard is only there when a speech model transcribed it.
        st.set_heard(
            match &f.text {
                Some(t) if !t.trim().is_empty() => t.trim().to_string(),
                Some(_) => "No words recognised".to_string(),
                None => String::new(),
            }
            .into(),
        );
        super::APP.with(|a| {
            if let Some(app) = a.borrow().as_ref().cloned() {
                app.setup_resume_meter();
            }
        });
        true
    }

    // ---------- Step 4: try it ----------

    /// A dictation finished while the setup window is in front on the "Try it" step: the text goes straight into the
    /// practice box (like the Notes page) instead of being pasted. Returns true when it took the text.
    pub(super) fn deliver_to_setup(&self, text: &str) -> bool {
        if !self.setup.is_open() {
            return false;
        }
        let st = self.setup.state();
        if st.get_step() != 4 || !win::hwnd(self.setup.win.window()).is_some_and(win::is_foreground) {
            return false;
        }
        let next = append_practice(st.get_practice_text().as_str(), text);
        st.set_practice_text(next.into());
        true
    }

    /// Smart cleanup's state for the "Try it" card and the Ready summary.
    pub(super) fn refresh_setup_cleanup(&self, c: &CleanupStatus) {
        let st = self.setup.state();
        st.set_cleanup_on(c.enabled);
        st.set_cleanup_badge(format!("Optional · SpeakoFlow Mini, ~{:.1} GB", c.size.bytes() as f64 / 1e9).into());
        let (note, summary) = if !c.enabled {
            ("Off by default.".to_string(), "Off".to_string())
        } else {
            match &c.state {
                LlmState::Downloading(p) => (format!("Downloading… {} %", percent(*p)), format!("On · downloading {} %", percent(*p))),
                LlmState::Error(_) => ("The download failed. You can retry it in Settings.".to_string(), "On · download failed".to_string()),
                LlmState::NotDownloaded => ("Starting the download…".to_string(), "On · not downloaded".to_string()),
                LlmState::Installed | LlmState::Starting | LlmState::Ready(_) => ("Ready.".to_string(), "On".to_string()),
            }
        };
        st.set_cleanup_note(note.into());
        st.set_summary_cleanup(summary.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_opens_only_for_a_foreground_launch_with_no_model_and_no_earlier_setup() {
        // (setup_done, has_model, background)
        assert_eq!(decide(false, false, false), Launch::Setup, "a new user");
        assert_eq!(decide(false, true, false), Launch::MarkDone, "an existing user with a model: silent");
        assert_eq!(decide(true, false, false), Launch::Nothing, "setup was done before, even with the models removed");
        assert_eq!(decide(true, true, false), Launch::Nothing);
        for (done, model) in [(false, false), (false, true), (true, false), (true, true)] {
            assert_eq!(decide(done, model, true), Launch::Nothing, "--background never shows setup (done {done}, model {model})");
        }
    }

    #[test]
    fn the_mic_test_calls_a_quiet_recording_too_quiet() {
        assert_eq!(verdict(0.4, 0.8), Verdict::Good);
        assert_eq!(verdict(0.06, 0.35), Verdict::Good, "just over both thresholds");
        assert_eq!(verdict(0.01, 0.8), Verdict::Quiet, "a nearly silent recording");
        assert_eq!(verdict(0.4, 0.1), Verdict::Quiet, "the meter never rose");
        assert_eq!(verdict(0.0, 0.0), Verdict::Quiet);
    }

    #[test]
    fn dictation_is_added_to_the_practice_box_after_what_is_there() {
        assert_eq!(append_practice("", "hello there"), "hello there");
        assert_eq!(append_practice("hello", " there "), "hello there");
        assert_eq!(append_practice("hello ", "there"), "hello there");
        assert_eq!(append_practice("hello\n", "there"), "hello\nthere");
        assert_eq!(append_practice("hello", "   "), "hello");
    }

    #[test]
    fn every_model_has_card_text() {
        for id in ["phonon-2.5-en", "phonon-2.5", "fihria-asr-0.6b", "fihria-asr-1.7b"] {
            let (name, _, langs, speed, acc) = about(id);
            assert!(name != "Speech model" && !langs.is_empty() && !speed.is_empty() && !acc.is_empty(), "{id}");
        }
        assert!(about("phonon-2.5-en").1 && !about("phonon-2.5").1, "only the English model is recommended");
    }
}
