use glam::DVec3;
use legion::{world::SubWorld, *};

use crate::{
    components::{miner::Miner, newton_body::NewtonBody, thruster::ThrusterBank},
    systems::energy::Energy,
    Dt,
};

/// Energy drained per unit of effort per second.
/// Effort is 0..1 per channel (linear + rotational), so max drain is 2× this.
pub const THRUSTER_DRAIN_RATE: f64 = 5.0;

/// Effort below this threshold is treated as zero (suppresses autopilot micro-correction drain).
const EFFORT_EPSILON: f64 = 1e-3;

pub fn apply_thrusters(body: &mut NewtonBody, bank: &mut ThrusterBank, dt: f64) {
    body.angular_vel += body.orientation * (bank.command.clamp_length_max(bank.max_rot_accel) * dt);
    bank.command = DVec3::ZERO;

    body.vel += body.orientation * (bank.linear_command.clamp_length_max(bank.max_lin_accel) * dt);
    bank.linear_command = DVec3::ZERO;
}

#[system]
#[write_component(NewtonBody)]
#[write_component(ThrusterBank)]
#[read_component(Miner)]
pub fn thruster(world: &mut SubWorld, #[resource] dt: &Dt, #[resource] energy: &mut Energy) {
    let dt = dt.0;

    // Non-miner entities thrust freely (no energy cost).
    let mut other_q = <(&mut NewtonBody, &mut ThrusterBank)>::query().filter(!component::<Miner>());
    for (body, bank) in other_q.iter_mut(world) {
        apply_thrusters(body, bank, dt);
    }

    // Miner thrust is energy-gated: calculate cost from commands, apply only if affordable.
    let mut miner_q = <(&Miner, &mut NewtonBody, &mut ThrusterBank)>::query();
    for (_, body, bank) in miner_q.iter_mut(world) {
        let lin = (bank.linear_command.length() / bank.max_lin_accel).min(1.0);
        let rot = (bank.command.length() / bank.max_rot_accel).min(1.0);
        let effort = lin + rot;
        let cost = effort * THRUSTER_DRAIN_RATE * dt;

        if effort <= EFFORT_EPSILON || energy.current >= cost {
            energy.current = (energy.current - cost).max(0.0);
            apply_thrusters(body, bank, dt);
        } else {
            energy.current = 0.0;
            bank.linear_command = DVec3::ZERO;
            bank.command = DVec3::ZERO;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        components::thruster::ThrusterBank,
        test_utils::{make_bank, make_body},
    };

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
