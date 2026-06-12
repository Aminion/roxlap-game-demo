use std::time::{Duration, Instant};

use legion::*;

use crate::Dt;

const PERIOD: Duration = Duration::from_secs(1);

pub struct PerformanceInfo {
    pub fps: u64,
    /// Raw values written by the render system each frame.
    pub frame_time_us_raw: u64,
    pub opticast_us_raw: u64,
    pub upload_us_raw: u64,
    /// Smoothed values shown in the overlay (updated once per second).
    pub frame_time_us: u64,
    pub opticast_us: u64,
    pub upload_us: u64,
    update_timer: Instant,
}

impl PerformanceInfo {
    pub fn new() -> Self {
        Self {
            fps: 0,
            frame_time_us_raw: 0,
            opticast_us_raw: 0,
            upload_us_raw: 0,
            frame_time_us: 0,
            opticast_us: 0,
            upload_us: 0,
            update_timer: Instant::now(),
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
    info.frame_time_us_raw = (dt.0 * 1_000_000.0) as u64;
    if info.update_timer.elapsed() >= PERIOD {
        info.fps = dt.0.recip() as u64;
        info.frame_time_us = info.frame_time_us_raw;
        info.opticast_us = info.opticast_us_raw;
        info.upload_us = info.upload_us_raw;
        info.update_timer = Instant::now();
    }
}
