use std::time::{Duration, Instant};

use legion::*;

use crate::Dt;

const PERIOD: Duration = Duration::from_secs(1);

pub struct PerformanceInfo {
    pub fps: u64,
    /// Per-frame work time in µs, written by the render system each frame.
    pub work_time_us_raw: u64,
    /// Work time shown in the overlay, sampled from raw once per second.
    pub work_time_us_display: u64,
    update_timer: Instant,
    /// Reset after render completes; elapsed is read at next render start.
    pub work_timer: Instant,
}

impl PerformanceInfo {
    pub fn new() -> Self {
        Self {
            fps: 0,
            work_time_us_raw: 0,
            work_time_us_display: 0,
            update_timer: Instant::now(),
            work_timer: Instant::now(),
        }
    }
}

impl Default for PerformanceInfo {
    fn default() -> Self {
        Self::new()
    }
}

#[system]
pub fn update_info(#[resource] dt: &Dt, #[resource] info: &mut PerformanceInfo) {
    if info.update_timer.elapsed() >= PERIOD {
        info.fps = dt.0.recip() as u64;
        info.work_time_us_display = info.work_time_us_raw;
        info.update_timer = Instant::now();
    }
}
