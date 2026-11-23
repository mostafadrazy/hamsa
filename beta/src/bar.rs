//! The dictation bar window: where it sits, showing and hiding without taking focus, dragging, the right-click menu.
//!
//! The window has a fixed size that fits the biggest state, so it never resizes (resizing flickers). A window region
//! limits it to what is shown (Slint reports `want-width` / `want-height`), so the hover hint only appears once the
//! pointer reaches the pill, and nothing around the bar blocks clicks.
//!
//! Saved positions use the same numbers as the Tauri build (top-left of its 360 × 72 bar window), so both builds put
//! the bar in the same spot.

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::time::Duration;

use hamsa_core::{BarPhase, Layout, Placement, PrefsStore};
use slint::{ComponentHandle, Timer, TimerMode};
use windows_sys::Win32::Foundation::{HWND, RECT};

use crate::win;
use crate::BarWindow;

/// Gap between the pill and the screen edge or taskbar, in logical px.
const GAP: f64 = 6.0;
/// The fixed window size in logical px (`width` / `height` in bar.slint).
const WIN_W: f64 = 400.0;
const WIN_H: f64 = 420.0;
/// Margin the Slint layout keeps around the pill (`padding` in bar.slint).
const PAD: f64 = 6.0;
/// The Tauri build's bar window, for converting saved positions.
const OLD_W: f64 = 360.0;
const OLD_H: f64 = 72.0;
const OLD_PAD: f64 = 8.0;
/// How long the idle bar stays up so its morph back to the resting pill is seen.
const HIDE_DELAY: Duration = Duration::from_millis(400);

/// Where the pill is pinned: its horizontal centre, and either its bottom edge (bar opens upward) or its top edge.
#[derive(Clone, Copy, Debug)]
struct Anchor {
    cx: i32,
    edge: i32,
    bottom: bool,
    scale: f64,
    work: (i32, i32, i32, i32),
}

pub struct Bar {
    pub win: BarWindow,
    prefs: Arc<PrefsStore>,
    hwnd: Cell<Option<HWND>>,
    phase: Cell<BarPhase>,
    menu: Cell<bool>,
    /// Pointer and window position when a drag started.
    drag_from: Cell<Option<((i32, i32), (i32, i32))>>,
    drag_timer: Timer,
    menu_timer: Timer,
    start_timer: Timer,
    /// Delays hiding the idle bar so the morph back to the resting pill can finish.
    hide_timer: Timer,
    anchor: RefCell<Option<Anchor>>,
}

impl Bar {
    pub fn new(win: BarWindow, prefs: Arc<PrefsStore>) -> std::rc::Rc<Self> {
        std::rc::Rc::new(Self {
            win,
            prefs,
            hwnd: Cell::new(None),
            phase: Cell::new(BarPhase::Idle),
            menu: Cell::new(false),
            drag_from: Cell::new(None),
            drag_timer: Timer::default(),
            menu_timer: Timer::default(),
            start_timer: Timer::default(),
            hide_timer: Timer::default(),
            anchor: RefCell::new(None),
        })
    }

    fn layout(&self) -> Layout {
        self.prefs.get().bar
    }

    /// Show the window once (so it exists), make it a focus-free overlay, then place it.
    /// Slint creates native windows only once the event loop runs, so finish when the handle appears.
    pub fn start(self: &std::rc::Rc<Self>) {
        self.sync_motion();
        let _ = self.win.show();
        self.sync_menu_state();
        let me = std::rc::Rc::downgrade(self);
        self.start_timer.start(TimerMode::Repeated, Duration::from_millis(20), move || {
            let Some(me) = me.upgrade() else { return };
            if let Some(h) = win::hwnd(me.win.window()) {
                me.start_timer.stop();
                win::make_overlay(h);
                me.hwnd.set(Some(h));
                me.place();
                me.apply_visibility(false);
            }
        });
    }

    fn anchor_for(&self, l: &Layout) -> Option<Anchor> {
        let work = |m: &win::Monitor| (m.work.left, m.work.top, m.work.right, m.work.bottom);
        if let Placement::Custom { x, y } = l.placement {
            // Only while the saved spot is still on a screen (a monitor may have been unplugged).
            if let Some(m) = win::monitor_at(x + 10, y + 10) {
                let s = m.scale;
                let cx = x + (OLD_W / 2.0 * s) as i32;
                let pill_mid = y + (OLD_H / 2.0 * s) as i32;
                let bottom = pill_mid > (m.work.top + m.work.bottom) / 2;
                let edge = if bottom { y + ((OLD_H - OLD_PAD) * s) as i32 } else { y + (OLD_PAD * s) as i32 };
                return Some(Anchor { cx, edge, bottom, scale: s, work: work(&m) });
            }
        }
        let m = l.monitor.as_deref().and_then(win::monitor_named).or_else(win::primary_monitor)?;
        let s = m.scale;
        let cx = (m.work.left + m.work.right) / 2;
        let gap = (GAP * s) as i32;
        Some(if matches!(l.placement, Placement::Top) {
            Anchor { cx, edge: m.work.top + gap, bottom: false, scale: s, work: work(&m) }
        } else {
            Anchor { cx, edge: m.work.bottom - gap, bottom: true, scale: s, work: work(&m) }
        })
    }

    /// The fixed window size in physical px.
    fn window_size(scale: f64) -> (i32, i32) {
        ((WIN_W * scale).round() as i32, (WIN_H * scale).round() as i32)
    }

    /// The content (what is drawn) as (left, top, right, bottom) in window px: bottom-centre, or top-centre.
    fn content_rect(&self, scale: f64, bottom: bool) -> (i32, i32, i32, i32) {
        let (ww, wh) = Self::window_size(scale);
        let w = ((self.win.get_want_width() as f64 * scale).round() as i32).clamp(1, ww);
        let h = ((self.win.get_want_height() as f64 * scale).round() as i32).clamp(1, wh);
        let x0 = (ww - w) / 2;
        let y0 = if bottom { wh - h } else { 0 };
        (x0, y0, x0 + w, y0 + h)
    }

    /// The content in screen px, from where the window is now.
    fn content_on_screen(&self) -> Option<RECT> {
        let r = win::window_rect(self.hwnd.get()?)?;
        let a = (*self.anchor.borrow())?;
        let (l, t, rt, b) = self.content_rect(a.scale, a.bottom);
        Some(RECT { left: r.left + l, top: r.top + t, right: r.left + rt, bottom: r.top + b })
    }

    /// Make only the content clickable (the rest of the window lets clicks through).
    fn update_region(&self) {
        let (Some(h), Some(a)) = (self.hwnd.get(), *self.anchor.borrow()) else { return };
        let (l, t, r, b) = self.content_rect(a.scale, a.bottom);
        win::set_region(h, l, t, r, b);
    }

    /// Pin the content to the anchor. The window itself keeps its size and only moves when the anchor does.
    pub fn place(&self) {
        let Some(h) = self.hwnd.get() else { return };
        if self.drag_from.get().is_none() {
            let Some(a) = self.anchor_for(&self.layout()) else { return };
            *self.anchor.borrow_mut() = Some(a);
            self.win.set_anchor_bottom(a.bottom);
            let (ww, wh) = Self::window_size(a.scale);
            let (cl, ct, cr, cb) = self.content_rect(a.scale, a.bottom);
            let pad = (PAD * a.scale) as i32;
            let (l, t, r, b) = a.work;
            // Where the window goes so the content is where the pill belongs, kept inside the work area.
            let x = (a.cx - ww / 2).clamp(l - cl, (r - (cr - cl) - cl).max(l - cl));
            let y = (if a.bottom { a.edge + pad - wh } else { a.edge - pad }).clamp(t - ct, (b - (cb - ct) - ct).max(t - ct));
            if win::window_rect(h).is_none_or(|w| (w.left, w.top, w.right - w.left, w.bottom - w.top) != (x, y, ww, wh)) {
                win::set_rect(h, x, y, ww, wh);
            }
        }
        self.update_region();
    }

    /// Show at once; hide after the morph back to the resting pill (`delay`), unless the state changes again first.
    fn apply_visibility(&self, delay: bool) {
        let Some(h) = self.hwnd.get() else { return };
        let show = self.phase.get() != BarPhase::Idle || self.menu.get() || self.layout().visible;
        // Every state change comes through here, so a pending hide is simply replaced.
        self.hide_timer.stop();
        if show || !delay {
            win::show_no_activate(h, show);
        } else {
            self.hide_timer.start(TimerMode::SingleShot, HIDE_DELAY, move || win::show_no_activate(h, false));
        }
    }

    /// Follow Windows' "Animation effects" setting.
    fn sync_motion(&self) {
        // HAMSA_BETA_REDUCE_MOTION is a dev switch to check the no-motion path without touching Windows settings.
        self.win.set_reduce_motion(!win::animations_enabled() || std::env::var_os("HAMSA_BETA_REDUCE_MOTION").is_some());
    }

    pub fn set_phase(&self, phase: BarPhase, level: f32) {
        self.win.set_level(level);
        if self.phase.replace(phase) == phase {
            return;
        }
        self.sync_motion();
        self.win.set_phase(match phase {
            BarPhase::Idle => 0,
            BarPhase::Listening => 1,
            BarPhase::Transcribing => 2,
        });
        if phase != BarPhase::Idle && self.menu.get() {
            self.close_menu();
        }
        if phase == BarPhase::Idle {
            self.win.set_stable("".into());
            self.win.set_tentative("".into());
        }
        self.apply_visibility(true);
    }

    /// Hands-free recording on or off: the pill shows a lock while it is on.
    pub fn set_locked(&self, locked: bool) {
        self.win.set_locked(locked);
    }

    pub fn set_preview(&self, stable: &str, tentative: &str) {
        self.win.set_stable(stable.into());
        self.win.set_tentative(tentative.into());
    }

    pub fn set_shortcut(&self, labels: &[String]) {
        let model: Vec<slint::SharedString> = labels.iter().map(|s| s.as_str().into()).collect();
        self.win.set_shortcut(slint::ModelRc::new(slint::VecModel::from(model)));
    }

    fn sync_menu_state(&self) {
        let l = self.layout();
        self.win.set_preview_on(l.preview);
        self.win.set_idle_on(l.visible);
        self.win.set_place_mode(match l.placement {
            Placement::Bottom => 0,
            Placement::Top => 1,
            Placement::Custom { .. } => 2,
        });
        self.win.set_last_text(hamsa_core::dictation::last_text().unwrap_or_default().into());
    }

    pub fn open_menu(self: &std::rc::Rc<Self>) {
        if self.menu.get() {
            self.close_menu();
            return;
        }
        self.sync_menu_state();
        self.sync_motion();
        self.menu.set(true);
        self.win.set_menu_open(true);
        self.apply_visibility(true);
        // The bar never has focus, so there's no "focus lost": close the menu on a click anywhere else.
        let me = std::rc::Rc::downgrade(self);
        let mut armed = false;
        self.menu_timer.start(TimerMode::Repeated, Duration::from_millis(30), move || {
            let Some(me) = me.upgrade() else { return };
            let down = win::any_button_down();
            // Wait for the right-click that opened the menu to be released first.
            if !armed {
                armed = !down;
                return;
            }
            if down {
                let inside = me.content_on_screen().is_some_and(|r| win::contains(&r, win::cursor()));
                if !inside {
                    me.close_menu();
                }
            }
        });
    }

    pub fn close_menu(&self) {
        self.menu_timer.stop();
        self.menu.set(false);
        self.win.set_menu_open(false);
        self.apply_visibility(true);
    }

    /// Follow the pointer until the left button is released, then save the spot.
    pub fn start_drag(self: &std::rc::Rc<Self>) {
        let Some(h) = self.hwnd.get() else { return };
        let Some(r) = win::window_rect(h) else { return };
        self.drag_from.set(Some((win::cursor(), (r.left, r.top))));
        let me = std::rc::Rc::downgrade(self);
        self.drag_timer.start(TimerMode::Repeated, Duration::from_millis(8), move || {
            let Some(me) = me.upgrade() else { return };
            let Some(((cx0, cy0), (x0, y0))) = me.drag_from.get() else { return };
            let Some(h) = me.hwnd.get() else { return };
            if !win::left_button_down() {
                me.drag_timer.stop();
                me.drag_from.set(None);
                me.save_dragged_position();
                return;
            }
            let (cx, cy) = win::cursor();
            win::move_to(h, x0 + cx - cx0, y0 + cy - cy0);
        });
    }

    fn save_dragged_position(&self) {
        let Some(r) = self.content_on_screen() else { return };
        let m = win::monitor_at((r.left + r.right) / 2, (r.top + r.bottom) / 2).or_else(win::primary_monitor);
        let s = m.as_ref().map_or(1.0, |m| m.scale);
        let bottom = self.anchor.borrow().map_or(true, |a| a.bottom);
        let pad = (PAD * s) as i32;
        let cx = (r.left + r.right) / 2;
        // Convert to the Tauri build's numbers (top-left of its 360 × 72 window).
        let x = cx - (OLD_W / 2.0 * s) as i32;
        let y = if bottom {
            let pill_bottom = r.bottom - pad;
            pill_bottom - ((OLD_H - OLD_PAD) * s) as i32
        } else {
            let pill_top = r.top + pad;
            pill_top - (OLD_PAD * s) as i32
        };
        self.prefs.update(|p| p.bar.placement = Placement::Custom { x, y });
        self.sync_menu_state();
        self.place();
    }

    /// Double-click: back to the bottom (or top) of the screen.
    pub fn reset_position(&self) {
        self.prefs.update(|p| {
            if matches!(p.bar.placement, Placement::Custom { .. }) {
                p.bar.placement = Placement::Bottom;
            }
            p.bar.monitor = None;
        });
        self.sync_menu_state();
        self.place();
    }

    /// From Settings or the menu. `position`: 0 bottom, 1 top, 2 keep where it was dragged.
    pub fn set_layout(&self, visible: bool, preview: bool, position: i32) {
        let current = self.layout().placement;
        let placement = match position {
            1 => Placement::Top,
            2 => match current {
                p @ Placement::Custom { .. } => p,
                _ => self.current_custom().unwrap_or(Placement::Bottom),
            },
            _ => Placement::Bottom,
        };
        self.prefs.update(|p| {
            p.bar.visible = visible;
            p.bar.preview = preview;
            p.bar.placement = placement;
        });
        self.sync_menu_state();
        self.place();
        self.apply_visibility(true);
    }

    /// The bar's current spot as a saved position.
    fn current_custom(&self) -> Option<Placement> {
        let a = (*self.anchor.borrow())?;
        let x = a.cx - (OLD_W / 2.0 * a.scale) as i32;
        let y = if a.bottom { a.edge - ((OLD_H - OLD_PAD) * a.scale) as i32 } else { a.edge - (OLD_PAD * a.scale) as i32 };
        Some(Placement::Custom { x, y })
    }
}
