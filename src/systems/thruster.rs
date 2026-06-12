use glam::DVec3;
use legion::{world::SubWorld, *};

use crate::{
    components::{newton_body::NewtonBody, thruster::ThrusterBank},
    Dt,
};

pub fn apply_thrusters(body: &mut NewtonBody, bank: &mut ThrusterBank, dt: f64) {
    body.angular_vel += body.orientation * (bank.command.clamp_length_max(bank.max_rot_accel) * dt);
    bank.command = DVec3::ZERO;

    body.vel += body.orientation * (bank.linear_command.clamp_length_max(bank.max_lin_accel) * dt);
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

    // mass=1.0, radius=1.0, rot_force=0.6 N → max_rot_accel = 5×0.6/(1×1) = 3.0 rad/s²
    // lin_force=5.0 N → max_lin_accel = 5.0 m/s²
    fn make_bank() -> ThrusterBank {
        ThrusterBank::new(1.0, 1.0, 0.6, 5.0)
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
            bank.linear_command = dir * bank.max_lin_accel;
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
        bank.linear_command = DVec3::Y * bank.max_lin_accel;
        apply_thrusters(&mut body, &mut bank, 1.0);
        assert!(
            body.vel.z > 0.0,
            "body +Y should map to world +Z after rotation_x(π/2)"
        );
        assert!(body.vel.x.abs() < 1e-12);
    }
}
