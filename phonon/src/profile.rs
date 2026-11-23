//! Cheap cumulative stage timers (one atomic add per stage call), read by `phonon::profile_report`.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

pub const STAGES: [&str; 6] = ["subsampling", "feed-forward", "attention", "conv module", "linear (all)", "attention core"];
pub const SUB: usize = 0;
pub const FF: usize = 1;
pub const ATT: usize = 2;
pub const CONV: usize = 3;
pub const LINEAR: usize = 4;
pub const ATT_CORE: usize = 5;

static NANOS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

pub fn add(stage: usize, since: Instant) {
    NANOS[stage].fetch_add(since.elapsed().as_nanos() as u64, Relaxed);
}

/// (stage, milliseconds) since the last call, then resets.
pub fn report() -> Vec<(&'static str, f64)> {
    STAGES.iter().enumerate().map(|(i, name)| (*name, NANOS[i].swap(0, Relaxed) as f64 / 1e6)).collect()
}
