use glam::{DQuat, DVec3, Vec2};
use legion::{world::SubWorld, *};

use crate::{
    components::{miner::Miner, newton_body::NewtonBody, thruster::ThrusterBank},
    math::reject,
    AutopilotTarget, MouseDelta,
};

/// Proportional angular gain (rad/s per radian of error).
const STEER_GAIN: f64 = 4.0;
/// Maximum rotation speed the autopilot targets (rad/s).
const MAX_ANGULAR_SPEED: f64 = 3.0;
/// Mouse sensitivity when rotating the target direction (rad/pixel).
const MOUSE_SENSITIVITY: f64 = 0.003;
/// Below this heading error the autopilot switches to PD centering mode.
const DEAD_ZONE: f64 = 0.01;
/// Proportional gain inside dead zone: pulls heading toward target center (rad/s² per radian).
const DEAD_ZONE_KP: f64 = 1.0;
/// Derivative gain inside dead zone: damps residual spin (rad/s² per rad/s).
/// Set to 2·√KP for critical damping.
const DEAD_ZONE_KD: f64 = 2.0;
/// Near-zero threshold for angles and vector lengths.
const EPSILON: f64 = 1e-9;

/// Steer toward `target_dir` using bang-bang control with a deceleration profile.
/// The desired angular speed is capped at √(2·a·angle) — the fastest speed that
/// still allows stopping at the target — so braking starts early enough to avoid overshoot.
/// Within DEAD_ZONE the autopilot only damps residual spin to suppress idle chatter.
pub fn apply_autopilot(body: &NewtonBody, bank: &mut ThrusterBank, target_dir: DVec3) {
    let ship_fwd = body.orientation * DVec3::NEG_Z;

    let steer_cross = ship_fwd.cross(target_dir);
    let steer_sin = steer_cross.length();
    let steer_cos = ship_fwd.dot(target_dir);
    let steer_angle = steer_sin.atan2(steer_cos);
    if steer_angle < EPSILON {
        return;
    }

    let max_a = bank.max_rot_accel;

    let steer_axis = if steer_sin > EPSILON {
        steer_cross / steer_sin
    } else {
        let alt = if ship_fwd.x.abs() < 0.9 {
            DVec3::X
        } else {
            DVec3::Y
        };
        ship_fwd.cross(alt).normalize()
    };

    // Strip roll (rotation around ship_fwd / body NEG_Z) from angular_vel so Q/E roll
    // doesn't interfere with autopilot steering. Roll doesn't change heading anyway.
    let heading_av = reject(body.angular_vel, ship_fwd);

    if steer_angle < DEAD_ZONE {
        // PD controller: P pulls toward target center, D damps heading spin only.
        let p_world = steer_axis * (steer_angle * DEAD_ZONE_KP);
        let d_world = -heading_av * DEAD_ZONE_KD;
        bank.command += body.orientation.inverse() * (p_world + d_world);
        return;
    }

    // Max speed that still allows stopping at the target under full deceleration.
    let safe_speed = (2.0 * max_a * steer_angle).sqrt();
    let desired_speed = safe_speed.min((steer_angle * STEER_GAIN).min(MAX_ANGULAR_SPEED));

    let desired_world = steer_axis * desired_speed;
    let error = desired_world - heading_av;
    if error.length() < EPSILON {
        return;
    }
    bank.command += body.orientation.inverse() * (error / error.length() * max_a);
}

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[write_component(ThrusterBank)]
pub fn autopilot(
    world: &mut SubWorld,
    #[resource] autopilot_target: &mut AutopilotTarget,
    #[resource] mouse_delta: &MouseDelta,
) {
    if *mouse_delta != Vec2::ZERO {
        let (cam_right, cam_up) = {
            let mut q = <(&Miner, &NewtonBody)>::query();
            let (_, body) = q.iter(world).next().expect("miner missing");
            (body.orientation * DVec3::X, body.orientation * DVec3::Y)
        };
        let delta = mouse_delta.as_dvec2() * (-MOUSE_SENSITIVITY);
        let yaw_rot = DQuat::from_axis_angle(cam_up, delta.x);
        let pitch_rot = DQuat::from_axis_angle(cam_right, delta.y);
        autopilot_target.0 = (yaw_rot * pitch_rot * autopilot_target.0).normalize();
    }

    let target_dir = autopilot_target.0;

    let mut q = <(&Miner, &NewtonBody, &mut ThrusterBank)>::query();
    let (_, body, bank) = q.iter_mut(world).next().expect("miner missing");
    apply_autopilot(body, bank, target_dir);
}

#[cfg(test)]
mod tests {
    use super::apply_autopilot;
    use crate::{
        components::{newton_body::NewtonBody, thruster::ThrusterBank},
        math::reject,
        systems::thruster::apply_thrusters,
        Dt,
    };
    use glam::{DQuat, DVec3};
    use proptest::prelude::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn dir(yaw: f64, pitch: f64) -> DVec3 {
        DVec3::new(
            pitch.cos() * yaw.sin(),
            pitch.sin(),
            -pitch.cos() * yaw.cos(),
        )
        .normalize()
    }

    fn simulate(mut body: NewtonBody, target: DVec3, seconds: f64, dt: f64) -> NewtonBody {
        let dt_obj = Dt(dt);
        for _ in 0..(seconds / dt) as usize {
            let mut bank = ThrusterBank::new(1.0, 1.0, 0.6, 5.0);
            apply_autopilot(&body, &mut bank, target);
            apply_thrusters(&mut body, &mut bank, dt);
            body.integrate_rotation(&dt_obj);
        }
        body
    }

    /// Run the bang-bang phase from `start_angle` and return the heading angular
    /// velocity magnitude the moment the ship first enters the dead zone.
    fn measure_entry_omega(start_angle: f64, dt: f64) -> Option<f64> {
        let target = dir(start_angle, 0.0);
        let mut body = NewtonBody {
            mass: 1.0,
            pos: DVec3::ZERO,
            vel: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_vel: DVec3::ZERO,
        };
        for _ in 0..(60.0 / dt) as usize {
            let mut bank = ThrusterBank::new(1.0, 1.0, 0.6, 5.0);
            apply_autopilot(&body, &mut bank, target);
            apply_thrusters(&mut body, &mut bank, dt);
            body.integrate_rotation(&Dt(dt));
            let heading = body.orientation * DVec3::NEG_Z;
            let angle = heading.dot(target).clamp(-1.0, 1.0).acos();
            if angle < super::DEAD_ZONE {
                let ship_fwd = body.orientation * DVec3::NEG_Z;
                return Some(reject(body.angular_vel, ship_fwd).length());
            }
        }
        None
    }

    /// 1-D Euler simulation of dead-zone PD settling.
    ///
    /// `init_e`     — signed angle error (positive: ship offset from center)
    /// `init_omega` — angular velocity toward center (positive = decreasing error)
    ///
    /// Returns ISE (integral of e² dt) over 3 s.  Exits early with a large
    /// penalty if the angle leaves the dead zone, matching real controller
    /// behaviour (bang-bang re-engages outside the dead zone so PD settling
    /// that immediately overshoots is a failure mode, not a free ride).
    fn dead_zone_cost(kp: f64, kd: f64, init_e: f64, init_omega: f64, dt: f64) -> f64 {
        // max_rot_accel for ThrusterBank::new(1.0, 1.0, 0.6, 5.0): 5*0.6/(1.0*1.0) = 3.0
        const MAX_ROT_ACCEL: f64 = 3.0;
        let mut e = init_e;
        let mut omega = init_omega;
        let mut ise = 0.0_f64;
        for _ in 0..(3.0 / dt) as usize {
            let cmd = (kp * e - kd * omega).clamp(-MAX_ROT_ACCEL, MAX_ROT_ACCEL);
            omega += cmd * dt;
            e -= omega * dt;
            if e.abs() > super::DEAD_ZONE {
                return 10.0; // exited dead zone: bang-bang re-engages, PD failed
            }
            ise += e * e * dt;
        }
        ise
    }

    fn heading_error(body: &NewtonBody, target: DVec3) -> f64 {
        let heading = body.orientation * DVec3::NEG_Z;
        heading.dot(target).clamp(-1.0, 1.0).acos()
    }

    // ── No NaN / inf in command under arbitrary single-step inputs ──────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]
        #[test]
        fn no_nan_or_inf(
            tgt_yaw   in -PI..PI,
            tgt_pitch in -FRAC_PI_2..FRAC_PI_2,
            ang_x in -10.0f64..10.0,
            ang_y in -10.0f64..10.0,
            ang_z in -10.0f64..10.0,
        ) {
            let target = dir(tgt_yaw, tgt_pitch);
            let body = NewtonBody {
                mass: 1.0,
                pos: DVec3::ZERO,
                vel: DVec3::ZERO,
                orientation: DQuat::IDENTITY,
                angular_vel: DVec3::new(ang_x, ang_y, ang_z),
            };
            let mut bank = ThrusterBank::new(1.0, 1.0, 0.6, 5.0);
            apply_autopilot(&body, &mut bank, target);
            prop_assert!(bank.command.is_finite(), "command NaN/inf");
        }
    }

    // ── Roll must not be damped when autopilot is active ──────────────────

    #[test]
    fn roll_not_damped_by_autopilot() {
        // Ship is inside the dead zone (steer_angle ≈ 0.005 rad < DEAD_ZONE = 0.01).
        // Holding Q (roll CW) must build up freely; the dead-zone D-term must not
        // oppose it.  Old code capped roll at max_accel / DEAD_ZONE_KD = 1.5 rad/s.
        let target = DVec3::new(0.005, 0.0, -1.0).normalize();
        let mut body = NewtonBody {
            mass: 1.0,
            pos: DVec3::ZERO,
            vel: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_vel: DVec3::ZERO,
        };
        let dt = 1.0 / 60.0;
        let dt_obj = Dt(dt);
        for _ in 0..60 {
            let mut bank = ThrusterBank::new(1.0, 1.0, 0.6, 5.0);
            bank.command += DVec3::NEG_Z * bank.max_rot_accel;
            apply_autopilot(&body, &mut bank, target);
            apply_thrusters(&mut body, &mut bank, dt);
            body.integrate_rotation(&dt_obj);
        }
        assert!(
            body.angular_vel.length() > 2.0,
            "autopilot damped roll: |ω|={:.3} rad/s (expected > 2.0)",
            body.angular_vel.length()
        );
    }

    #[test]
    fn heading_converges_while_rolling() {
        // Holding Q throughout must not prevent the autopilot from converging
        // the heading to the target.
        let target = dir(0.8, 0.3);
        let mut body = NewtonBody {
            mass: 1.0,
            pos: DVec3::ZERO,
            vel: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_vel: DVec3::ZERO,
        };
        let dt = 1.0 / 60.0;
        let dt_obj = Dt(dt);
        for _ in 0..(8.0 / dt) as usize {
            let mut bank = ThrusterBank::new(1.0, 1.0, 0.6, 5.0);
            bank.command += DVec3::NEG_Z * bank.max_rot_accel;
            apply_autopilot(&body, &mut bank, target);
            apply_thrusters(&mut body, &mut bank, dt);
            body.integrate_rotation(&dt_obj);
        }
        let err = heading_error(&body, target);
        assert!(
            err < 0.05,
            "heading failed to converge while rolling: err={:.4} rad",
            err
        );
    }

    // ── Heading must converge to target within 5 s ─────────────────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]
        #[test]
        fn heading_converges(
            tgt_yaw   in -PI..PI,
            tgt_pitch in -FRAC_PI_2..FRAC_PI_2,
        ) {
            let target = dir(tgt_yaw, tgt_pitch);
            let body = NewtonBody {
                mass: 1.0,
                pos: DVec3::ZERO,
                vel: DVec3::ZERO,
                orientation: DQuat::IDENTITY,
                angular_vel: DVec3::ZERO,
            };
            let err = heading_error(&simulate(body, target, 5.0, 1.0 / 60.0), target);
            prop_assert!(
                err < 0.05,
                "heading_error={:.4} rad after 5 s; tgt_pitch={:.3}",
                err, tgt_pitch,
            );
        }
    }

    /// Grid search over (KP, KD) to find dead-zone gains that minimize ISE
    /// across realistic dt values and entry velocities produced by bang-bang.
    /// Run with: cargo test tune_pd -- --ignored --nocapture
    #[test]
    #[ignore]
    fn tune_pd() {
        // ── Phase 1: collect actual entry omegas from bang-bang handoff ───────
        let measure_dts = [1.0 / 120.0, 1.0 / 60.0, 1.0 / 30.0, 1.0 / 15.0];
        let start_angles = [0.05_f64, 0.3, 1.0, PI - 0.01];
        let mut entry_omegas = vec![0.0_f64]; // zero covers "started inside dead zone"
        for &dt in &measure_dts {
            for &sa in &start_angles {
                if let Some(omega) = measure_entry_omega(sa, dt) {
                    entry_omegas.push(omega);
                }
            }
        }
        // Only keep omegas the PD can physically settle before the dead zone
        // boundary: v²/(2·a) ≤ DEAD_ZONE  →  v ≤ √(2·MAX_ROT_ACCEL·DEAD_ZONE).
        // Higher entries always trigger the exit penalty regardless of gains,
        // adding uniform noise rather than discriminating between gain choices.
        const MAX_SETTLEABLE_OMEGA: f64 =
            // sqrt(2 * 3.0 * DEAD_ZONE); computed as const-compatible literal
            0.2449; // ≈ sqrt(2 * 3.0 * 0.01)
        entry_omegas.retain(|&omega| omega <= MAX_SETTLEABLE_OMEGA);
        entry_omegas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        entry_omegas.dedup_by(|a, b| (*a - *b).abs() < 0.003);
        println!(
            "Handoff omega range: {:.4}..={:.4} rad/s  ({} settleable samples)",
            entry_omegas.first().unwrap(),
            entry_omegas.last().unwrap(),
            entry_omegas.len(),
        );

        // ── Phase 2: grid search ──────────────────────────────────────────────
        let eval_dts = measure_dts;
        let init_angles = [0.001_f64, 0.003, 0.005, 0.007, 0.009];
        let n_scenarios = eval_dts.len() * init_angles.len() * entry_omegas.len();

        let avg_cost = |kp: f64, kd: f64| -> f64 {
            let mut total = 0.0;
            for &dt in &eval_dts {
                for &angle in &init_angles {
                    for &omega in &entry_omegas {
                        total += dead_zone_cost(kp, kd, angle, omega, dt);
                    }
                }
            }
            total / n_scenarios as f64
        };

        // KP: 0.1..=10.0 in 0.1 steps; KD: 0.1..=10.0 in 0.1 steps
        let kp_steps: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
        let kd_steps: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();

        let (best_kp, best_kd, best_cost) = kp_steps
            .iter()
            .flat_map(|&kp| kd_steps.iter().map(move |&kd| (kp, kd)))
            .map(|(kp, kd)| (kp, kd, avg_cost(kp, kd)))
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();

        let current_cost = avg_cost(super::DEAD_ZONE_KP, super::DEAD_ZONE_KD);
        let improvement = (current_cost - best_cost) / current_cost * 100.0;

        println!(
            "Current : KP={:.2} KD={:.2}  avg_ISE={:.6e}",
            super::DEAD_ZONE_KP,
            super::DEAD_ZONE_KD,
            current_cost,
        );
        println!(
            "Best    : KP={:.2} KD={:.2}  avg_ISE={:.6e}  ({:+.1}%)",
            best_kp, best_kd, best_cost, -improvement,
        );
        if improvement > 5.0 {
            println!(
                "→ Update DEAD_ZONE_KP = {:.2} and DEAD_ZONE_KD = {:.2}",
                best_kp, best_kd,
            );
        } else {
            println!("→ Current gains are near-optimal (< 5% improvement available).");
        }
    }
}
