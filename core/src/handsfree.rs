//! Tap / double-tap / hold logic for the dictation shortcut, as a pure state machine (no keys, no clock: every input
//! carries its own timestamp), so it can be unit-tested.
//!
//! Recording starts on the first press, exactly as before, so no audio is lost. What happens next depends on how
//! the key is used:
//!
//! - Hold: released after `TAP_MAX` or more -> `Finish` (paste). Nothing new.
//! - Tap: released sooner -> the recording keeps running for `DOUBLE_TAP_WINDOW`, waiting for a second tap.
//!   - No second press in time -> `Discard` (dropped silently, no error, no history entry).
//!   - A second press in time -> `Lock`: hands-free, the recording stays on without holding.
//! - Locked: a press of the shortcut -> `Finish`. Esc -> `Cancel`. After `MAX_LOCKED` -> `Finish`.
//!
//! With hands-free switched off a release always means `Finish`, as in plain hold-to-talk.

use std::time::{Duration, Instant};

/// A press shorter than this is a tap, not a hold.
pub const TAP_MAX: Duration = Duration::from_millis(300);
/// The second tap must start within this long after the first one was released.
pub const DOUBLE_TAP_WINDOW: Duration = Duration::from_millis(400);
/// Safety cap so a stuck key can't record forever.
pub const MAX_HELD: Duration = Duration::from_secs(300);
/// Safety cap for hands-free recording.
pub const MAX_LOCKED: Duration = Duration::from_secs(600);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    /// The whole shortcut went down.
    Down,
    /// A shortcut key was released.
    Up,
    /// Esc was pressed (only matters while locked).
    Esc,
    /// Time passed (the dictation loop ticks every 50 ms while recording).
    Tick,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Start recording.
    Start,
    /// Throw the running recording away silently and start a new one (a late second tap).
    Restart,
    /// Hands-free is now on; the recording keeps running.
    Lock,
    /// Stop and process the recording.
    Finish,
    /// Stop and throw the recording away (Esc).
    Cancel,
    /// A lone tap: throw the recording away silently.
    Discard,
}

#[derive(Clone, Copy, Debug)]
enum State {
    Idle,
    /// The shortcut is down; `hands_free` is whether a short press may become a tap.
    Held { since: Instant, hands_free: bool },
    /// A short tap was released; waiting for a second one until `deadline`.
    Pending { deadline: Instant },
    /// Hands-free recording. `key_down`: the second press that locked it is still down.
    Locked { since: Instant, key_down: bool },
}

#[derive(Debug)]
pub struct Machine {
    state: State,
    enabled: bool,
}

impl Default for Machine {
    fn default() -> Self {
        Self::new()
    }
}

impl Machine {
    pub fn new() -> Self {
        Self { state: State::Idle, enabled: true }
    }

    /// Whether double-tap may lock; read when a press starts.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Hands-free recording is on.
    pub fn locked(&self) -> bool {
        matches!(self.state, State::Locked { .. })
    }

    /// A tap was released and the machine is waiting to see whether a second one follows.
    pub fn pending(&self) -> bool {
        matches!(self.state, State::Pending { .. })
    }

    /// Back to idle (the recording was ended some other way: Stop or Cancel button, another key pressed, error).
    pub fn reset(&mut self) {
        self.state = State::Idle;
    }

    pub fn step(&mut self, now: Instant, input: Input) -> Option<Action> {
        match (self.state, input) {
            (State::Idle, Input::Down) => {
                self.state = State::Held { since: now, hands_free: self.enabled };
                Some(Action::Start)
            }
            (State::Idle, _) => None,

            (State::Held { since, hands_free }, Input::Up) => {
                if hands_free && now.saturating_duration_since(since) < TAP_MAX {
                    self.state = State::Pending { deadline: now + DOUBLE_TAP_WINDOW };
                    None
                } else {
                    self.state = State::Idle;
                    Some(Action::Finish)
                }
            }
            (State::Held { since, .. }, Input::Tick) if now.saturating_duration_since(since) > MAX_HELD => {
                self.state = State::Idle;
                Some(Action::Finish)
            }
            (State::Held { .. }, _) => None,

            (State::Pending { deadline }, Input::Down) => {
                if now <= deadline {
                    self.state = State::Locked { since: now, key_down: true };
                    Some(Action::Lock)
                } else {
                    // The tick that would have dropped the first recording hasn't run yet.
                    self.state = State::Held { since: now, hands_free: self.enabled };
                    Some(Action::Restart)
                }
            }
            (State::Pending { deadline }, Input::Tick) if now > deadline => {
                self.state = State::Idle;
                Some(Action::Discard)
            }
            (State::Pending { .. }, _) => None,

            (State::Locked { key_down: false, .. }, Input::Down) => {
                self.state = State::Idle;
                Some(Action::Finish)
            }
            (State::Locked { since, .. }, Input::Up) => {
                self.state = State::Locked { since, key_down: false };
                None
            }
            (State::Locked { .. }, Input::Esc) => {
                self.state = State::Idle;
                Some(Action::Cancel)
            }
            (State::Locked { since, .. }, Input::Tick) if now.saturating_duration_since(since) > MAX_LOCKED => {
                self.state = State::Idle;
                Some(Action::Finish)
            }
            (State::Locked { .. }, _) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives a machine with times in milliseconds from a fixed start.
    struct Rig {
        m: Machine,
        t0: Instant,
    }

    impl Rig {
        fn new() -> Self {
            Self { m: Machine::new(), t0: Instant::now() }
        }
        fn at(&mut self, ms: u64, input: Input) -> Option<Action> {
            self.m.step(self.t0 + Duration::from_millis(ms), input)
        }
    }

    use Action::*;
    use Input::*;

    #[test]
    fn hold_starts_on_press_and_finishes_on_release() {
        let mut r = Rig::new();
        assert_eq!(r.at(0, Down), Some(Start));
        assert_eq!(r.at(50, Tick), None);
        assert_eq!(r.at(1500, Up), Some(Finish));
        assert!(!r.m.locked());
        assert_eq!(r.at(1600, Tick), None);
    }

    #[test]
    fn hold_just_over_the_tap_limit_is_a_hold() {
        let mut r = Rig::new();
        r.at(0, Down);
        assert_eq!(r.at(300, Up), Some(Finish));
    }

    #[test]
    fn lone_tap_is_discarded_after_the_window() {
        let mut r = Rig::new();
        assert_eq!(r.at(0, Down), Some(Start));
        assert_eq!(r.at(120, Up), None);
        assert!(r.m.pending());
        assert_eq!(r.at(300, Tick), None);
        assert_eq!(r.at(520, Tick), None, "exactly at the deadline it is still waiting");
        assert_eq!(r.at(571, Tick), Some(Discard));
        assert!(!r.m.pending());
        assert_eq!(r.at(650, Tick), None);
    }

    #[test]
    fn double_tap_locks() {
        let mut r = Rig::new();
        assert_eq!(r.at(0, Down), Some(Start));
        assert_eq!(r.at(100, Up), None);
        assert_eq!(r.at(300, Down), Some(Lock));
        assert!(r.m.locked());
        // Releasing the second press leaves it recording.
        assert_eq!(r.at(380, Up), None);
        assert!(r.m.locked());
        assert_eq!(r.at(5_000, Tick), None);
    }

    #[test]
    fn second_press_held_for_long_still_just_locks() {
        let mut r = Rig::new();
        r.at(0, Down);
        r.at(100, Up);
        assert_eq!(r.at(200, Down), Some(Lock));
        assert_eq!(r.at(4_000, Tick), None);
        assert_eq!(r.at(4_100, Up), None);
        assert!(r.m.locked());
    }

    #[test]
    fn tap_while_locked_finishes() {
        let mut r = Rig::new();
        r.at(0, Down);
        r.at(100, Up);
        r.at(250, Down);
        r.at(320, Up);
        assert_eq!(r.at(8_000, Down), Some(Finish));
        assert!(!r.m.locked());
        // Its release changes nothing.
        assert_eq!(r.at(8_090, Up), None);
        // And the next press starts a fresh recording.
        assert_eq!(r.at(9_000, Down), Some(Start));
    }

    #[test]
    fn esc_cancels_only_while_locked() {
        let mut r = Rig::new();
        assert_eq!(r.at(0, Esc), None);
        r.at(10, Down);
        assert_eq!(r.at(50, Esc), None, "while holding, Esc is not ours");
        r.at(100, Up);
        assert_eq!(r.at(150, Esc), None, "while waiting for a second tap neither");
        assert!(r.m.pending());
        r.at(200, Down);
        r.at(260, Up);
        assert_eq!(r.at(3_000, Esc), Some(Cancel));
        assert!(!r.m.locked());
        assert_eq!(r.at(3_100, Esc), None);
        assert_eq!(r.at(3_200, Tick), None);
    }

    #[test]
    fn esc_cancels_even_while_the_locking_press_is_still_down() {
        let mut r = Rig::new();
        r.at(0, Down);
        r.at(100, Up);
        r.at(200, Down);
        assert_eq!(r.at(900, Esc), Some(Cancel));
        assert_eq!(r.at(950, Up), None);
    }

    #[test]
    fn slow_second_tap_is_two_separate_taps() {
        let mut r = Rig::new();
        assert_eq!(r.at(0, Down), Some(Start));
        assert_eq!(r.at(100, Up), None);
        assert_eq!(r.at(550, Tick), Some(Discard));
        // The second tap comes after the window: a new recording that is itself a lone tap.
        assert_eq!(r.at(700, Down), Some(Start));
        assert_eq!(r.at(800, Up), None);
        assert!(!r.m.locked());
        assert_eq!(r.at(1_250, Tick), Some(Discard));
    }

    #[test]
    fn late_second_tap_before_the_tick_restarts() {
        let mut r = Rig::new();
        r.at(0, Down);
        r.at(100, Up);
        assert_eq!(r.at(501, Down), Some(Restart));
        assert!(!r.m.locked());
        assert_eq!(r.at(2_000, Up), Some(Finish), "the new press is a normal hold");
    }

    #[test]
    fn second_tap_right_at_the_edge_of_the_window_locks() {
        let mut r = Rig::new();
        r.at(0, Down);
        r.at(100, Up);
        assert_eq!(r.at(500, Down), Some(Lock));
    }

    #[test]
    fn very_long_hold_is_cut_off() {
        let mut r = Rig::new();
        r.at(0, Down);
        assert_eq!(r.at(299_000, Tick), None);
        assert_eq!(r.at(300_100, Tick), Some(Finish));
        assert_eq!(r.at(300_200, Up), None, "the late release does nothing");
    }

    #[test]
    fn very_long_hold_released_normally_finishes() {
        let mut r = Rig::new();
        r.at(0, Down);
        assert_eq!(r.at(120_000, Up), Some(Finish));
    }

    #[test]
    fn locked_recording_finishes_by_itself_after_ten_minutes() {
        let mut r = Rig::new();
        r.at(0, Down);
        r.at(100, Up);
        r.at(200, Down);
        r.at(260, Up);
        assert_eq!(r.at(200 + 599_000, Tick), None);
        assert!(r.m.locked());
        assert_eq!(r.at(200 + 600_050, Tick), Some(Finish));
        assert!(!r.m.locked());
    }

    #[test]
    fn with_hands_free_off_a_short_press_just_finishes() {
        let mut r = Rig::new();
        r.m.set_enabled(false);
        assert_eq!(r.at(0, Down), Some(Start));
        assert_eq!(r.at(100, Up), Some(Finish), "same as plain hold-to-talk (the recorder drops it as too short)");
        // A second quick press is just another dictation, never a lock.
        assert_eq!(r.at(200, Down), Some(Start));
        assert_eq!(r.at(250, Up), Some(Finish));
        assert!(!r.m.locked());
    }

    #[test]
    fn switching_the_setting_applies_from_the_next_press() {
        let mut r = Rig::new();
        r.at(0, Down);
        r.m.set_enabled(false);
        assert_eq!(r.at(100, Up), None, "this press began with hands-free on");
        r.at(150, Down);
        assert!(r.m.locked());
        r.m.set_enabled(false);
        assert_eq!(r.at(5_000, Down), None, "second press is still down");
    }

    #[test]
    fn reset_forgets_everything() {
        let mut r = Rig::new();
        r.at(0, Down);
        r.at(100, Up);
        r.m.reset();
        assert_eq!(r.at(600, Tick), None);
        assert_eq!(r.at(700, Down), Some(Start));
    }

    #[test]
    fn key_repeat_downs_are_ignored() {
        let mut r = Rig::new();
        assert_eq!(r.at(0, Down), Some(Start));
        assert_eq!(r.at(40, Down), None);
        assert_eq!(r.at(900, Up), Some(Finish));
    }
}
