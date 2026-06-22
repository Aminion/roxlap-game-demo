use glam::{DQuat, DVec3, Vec2};
use legion::{world::SubWorld, *};

use crate::{
    components::{
        camera::CameraComponent, miner::Miner, newton_body::NewtonBody, thruster::ThrusterBank,
    },
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
#[read_component(CameraComponent)]
#[read_component(NewtonBody)]
#[write_component(ThrusterBank)]
pub fn autopilot(
    world: &mut SubWorld,
    #[resource] autopilot_target: &mut AutopilotTarget,
    #[resource] mouse_delta: &MouseDelta,
) {
    if *mouse_delta != Vec2::ZERO {
        let (_, cam) = {
            let mut q = <(&Miner, &CameraComponent)>::query();
            q.iter(world).next().expect("miner missing")
        };
        let c = &cam.0;
        let cam_right = DVec3::from(c.right.map(f64::from));
        let cam_up = -DVec3::from(c.down.map(f64::from));
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

    fn simulate(mut body: NewtonBody, target: DVec3, seconds: f64) -> NewtonBody {
        let dt = 1.0 / 60.0;
        let dt_obj = Dt(dt);
        for _ in 0..(seconds / dt) as usize {
            let mut bank = ThrusterBank::new(1.0, 1.0, 0.6, 5.0);
            apply_autopilot(&body, &mut bank, target);
            apply_thrusters(&mut body, &mut bank, dt);
            body.integrate_rotation(&dt_obj);
        }
        body
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
            let err = heading_error(&simulate(body, target, 5.0), target);
            prop_assert!(
                err < 0.05,
                "heading_error={:.4} rad after 5 s; tgt_pitch={:.3}",
                err, tgt_pitch,
            );
        }
    }
}
