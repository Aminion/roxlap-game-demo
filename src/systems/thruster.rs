use glam::DVec3;
use legion::{world::SubWorld, *};

use crate::{
    components::{newton_body::NewtonBody, thruster::ThrusterBank},
    Dt,
};

const MIN_COMMAND: f64 = 1e-15;

pub fn apply_thrusters(body: &mut NewtonBody, bank: &mut ThrusterBank, dt: f64) {
    // --- Rotational ---
    let mag = bank.command.length();
    if mag >= MIN_COMMAND {
        let dir = bank.command / mag;
        let throttle = (mag / bank.max_accel(body.mass)).min(1.0);
        let accel = bank.accel_per_thruster(body.mass);
        for torque in bank.torques {
            let activation = dir.dot(torque).max(0.0) * throttle;
            body.angular_vel += body.orientation * (torque * (activation * accel * dt));
        }
    }
    bank.command = DVec3::ZERO;

    // --- Linear ---
    let lin_mag = bank.linear_command.length();
    if lin_mag >= MIN_COMMAND {
        let lin_dir = bank.linear_command / lin_mag;
        let max_la = bank.max_linear_accel(body.mass);
        let throttle = (lin_mag / max_la).min(1.0);
        let la = bank.linear_force / body.mass;
        for axis in bank.linear_axes {
            let activation = lin_dir.dot(axis).max(0.0) * throttle;
            body.vel += body.orientation * (axis * (activation * la * dt));
        }
    }
    bank.linear_command = DVec3::ZERO;
}

#[system]
#[write_component(NewtonBody)]
#[write_component(ThrusterBank)]
pub fn thruster(world: &mut SubWorld, #[resource] dt: &Dt) {
    let mut query = <(&mut NewtonBody, &mut ThrusterBank)>::query();
    for (body, bank) in query.iter_mut(world) {
        apply_thrusters(body, bank, dt.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::thruster::ThrusterBank;
    use glam::{DQuat, DVec3};

    fn make_body() -> NewtonBody {
        NewtonBody {
            mass: 1.0,
            pos: DVec3::ZERO,
            vel: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_vel: DVec3::ZERO,
        }
    }

    // radius=1.0, rot_force=0.6 N → accel_per = 0.6×1/(0.4×1×1) = 1.5 rad/s²
    // max_accel = 1.5 × 2 = 3.0 rad/s²  (same as old 0.3 N × 4 nozzles)
    // lin_force=5.0 N → max_linear_accel = 5.0 m/s²
    fn make_bank() -> ThrusterBank {
        ThrusterBank::new(1.0, 0.6, 5.0)
    }

    // ── Rotational ──────────────────────────────────────────────────────────

    #[test]
    fn command_zeroed_after_apply() {
        let mut body = make_body();
        let mut bank = make_bank();
        bank.command = DVec3::Z;
        apply_thrusters(&mut body, &mut bank, 1.0 / 60.0);
        assert_eq!(bank.command, DVec3::ZERO);
    }

    #[test]
    fn zero_command_leaves_body_unchanged() {
        let mut body = make_body();
        body.angular_vel = DVec3::new(1.0, 2.0, 3.0);
        let before = body.angular_vel;
        let mut bank = make_bank();
        apply_thrusters(&mut body, &mut bank, 1.0 / 60.0);
        assert_eq!(body.angular_vel, before);
    }

    #[test]
    fn angular_vel_moves_in_commanded_direction() {
        for dir in [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::NEG_X,
            DVec3::NEG_Y,
            DVec3::NEG_Z,
        ] {
            let mut body = make_body();
            let mut bank = make_bank();
            bank.command = dir * 3.0;
            apply_thrusters(&mut body, &mut bank, 1.0);
            let dot = body.angular_vel.dot(dir);
            assert!(
                dot > 0.5,
                "angular_vel not in commanded direction {dir:?}: dot={dot}"
            );
        }
    }

    #[test]
    fn rot_no_nan_or_inf() {
        let mut body = make_body();
        let mut bank = make_bank();
        bank.command = DVec3::new(0.3, -0.1, 0.7);
        apply_thrusters(&mut body, &mut bank, 1.0 / 60.0);
        assert!(body.angular_vel.is_finite());
    }

    // ── Linear ──────────────────────────────────────────────────────────────

    #[test]
    fn linear_command_zeroed_after_apply() {
        let mut body = make_body();
        let mut bank = make_bank();
        bank.linear_command = DVec3::Y;
        apply_thrusters(&mut body, &mut bank, 1.0 / 60.0);
        assert_eq!(bank.linear_command, DVec3::ZERO);
    }

    #[test]
    fn linear_vel_moves_in_commanded_direction() {
        for dir in [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::NEG_X,
            DVec3::NEG_Y,
            DVec3::NEG_Z,
        ] {
            let mut body = make_body();
            let mut bank = make_bank();
            bank.linear_command = dir * bank.max_linear_accel(body.mass);
            apply_thrusters(&mut body, &mut bank, 1.0);
            let dot = body.vel.dot(dir);
            assert!(
                dot > 0.5,
                "vel not in commanded direction {dir:?}: dot={dot}"
            );
        }
    }

    #[test]
    fn linear_thrust_respects_orientation() {
        use std::f64::consts::FRAC_PI_2;
        // rotation_x(π/2): body +Y → world +Z
        let mut body = make_body();
        body.orientation = DQuat::from_rotation_x(FRAC_PI_2);
        let mut bank = make_bank();
        bank.linear_command = DVec3::Y * bank.max_linear_accel(body.mass);
        apply_thrusters(&mut body, &mut bank, 1.0);
        assert!(
            body.vel.z > 0.0,
            "body +Y should map to world +Z after rotation_x(π/2)"
        );
        assert!(body.vel.x.abs() < 1e-12);
    }

}
