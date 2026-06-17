use std::collections::HashSet;

use glam::DVec3;
use legion::{world::SubWorld, *};

use crate::{
    components::{miner::Miner, newton_body::NewtonBody, thruster::ThrusterBank},
    input::PlayerInput,
};

/// Proportional retro-thrust/torque gain (s⁻¹). Terminal velocity ≈ max_lin_accel / DAMPING_GAIN.
const DAMPING_GAIN: f64 = 1.5;
const MIN_DIR_SQ: f64 = 1e-15;

fn axis(inputs: &HashSet<PlayerInput>, pos: PlayerInput, neg: PlayerInput) -> f64 {
    (inputs.contains(&pos) as i32 - inputs.contains(&neg) as i32) as f64
}

pub fn apply_miner_input(inputs: &HashSet<PlayerInput>, bank: &mut ThrusterBank) {
    let cw = inputs.contains(&PlayerInput::RollCW);
    let ccw = inputs.contains(&PlayerInput::RollCCW);
    if cw == ccw {
        return;
    }
    let sign: f64 = if cw { 1.0 } else { -1.0 };
    bank.command += DVec3::NEG_Z * (bank.max_rot_accel * sign);
}

/// Write a body-space linear-acceleration command for W/S (±Y) and A/D (±X).
/// Diagonal input is normalised; the thruster system converts it to world-space Δvel.
pub fn apply_miner_translation(inputs: &HashSet<PlayerInput>, bank: &mut ThrusterBank) {
    let local = DVec3::new(
        axis(inputs, PlayerInput::ThrustRight, PlayerInput::ThrustLeft),
        axis(inputs, PlayerInput::ThrustUp, PlayerInput::ThrustDown),
        axis(
            inputs,
            PlayerInput::ThrustBackward,
            PlayerInput::ThrustForward,
        ), // body -Z = nose/forward
    );

    if local.length_squared() > MIN_DIR_SQ {
        bank.linear_command += local.normalize() * bank.max_lin_accel;
    }
}

/// Write a retro-thrust command opposing body velocity (only when TAB is held).
/// Adds to linear_command alongside thrust; the bank caps at max_lin_accel.
/// At low speed this is proportional drag; at high speed it saturates (bang-bang).
pub fn apply_linear_damping(
    inputs: &HashSet<PlayerInput>,
    body: &NewtonBody,
    bank: &mut ThrusterBank,
) {
    if !inputs.contains(&PlayerInput::Damping) {
        return;
    }
    let vel_body = body.orientation.inverse() * body.vel;
    bank.linear_command += -vel_body * DAMPING_GAIN;
}

/// Write a counter-torque command opposing roll (only when TAB is held).
/// Only damps the roll component (angular_vel along the nose axis) so it
/// doesn't interfere with the autopilot's heading D-term.
pub fn apply_angular_damping(
    inputs: &HashSet<PlayerInput>,
    body: &NewtonBody,
    bank: &mut ThrusterBank,
) {
    if !inputs.contains(&PlayerInput::Damping) {
        return;
    }
    let ship_fwd = body.orientation * DVec3::NEG_Z;
    let roll_world = ship_fwd * body.angular_vel.dot(ship_fwd);
    bank.command += body.orientation.inverse() * (-roll_world * DAMPING_GAIN);
}

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[write_component(ThrusterBank)]
pub fn miner_input(world: &mut SubWorld, #[resource] inputs: &HashSet<PlayerInput>) {
    let mut query = <(&Miner, &NewtonBody, &mut ThrusterBank)>::query();
    for (_, body, bank) in query.iter_mut(world) {
        apply_miner_input(inputs, bank);
        apply_miner_translation(inputs, bank);
        apply_linear_damping(inputs, body, bank);
        apply_angular_damping(inputs, body, bank);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{newton_body::NewtonBody, thruster::ThrusterBank};
    use crate::systems::thruster::apply_thrusters;
    use glam::{DQuat, DVec3};
    use proptest::prelude::*;

    fn make_body() -> NewtonBody {
        NewtonBody {
            mass: 1.0,
            pos: DVec3::ZERO,
            vel: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_vel: DVec3::ZERO,
        }
    }

    fn make_bank() -> ThrusterBank {
        ThrusterBank::new(1.0, 1.0, 0.6, 5.0)
    }

    fn arb_player_input() -> impl Strategy<Value = PlayerInput> {
        prop_oneof![
            Just(PlayerInput::RollCW),
            Just(PlayerInput::RollCCW),
            Just(PlayerInput::ThrustUp),
            Just(PlayerInput::ThrustDown),
            Just(PlayerInput::ThrustLeft),
            Just(PlayerInput::ThrustRight),
            Just(PlayerInput::ThrustForward),
            Just(PlayerInput::ThrustBackward),
            Just(PlayerInput::Damping),
        ]
    }

    fn arb_inputs() -> impl Strategy<Value = HashSet<PlayerInput>> {
        prop::collection::hash_set(arb_player_input(), 0..=2)
    }

    // ── apply_miner_input ─────────────────────────────────────────────────

    #[test]
    fn no_input_bank_unchanged() {
        let mut bank = make_bank();
        apply_miner_input(&HashSet::new(), &mut bank);
        assert_eq!(bank.command, DVec3::ZERO);
    }

    #[test]
    fn opposite_roll_keys_cancel() {
        let mut both = HashSet::new();
        both.insert(PlayerInput::RollCW);
        both.insert(PlayerInput::RollCCW);
        let mut bank = make_bank();
        apply_miner_input(&both, &mut bank);
        assert_eq!(bank.command, DVec3::ZERO);
    }

    proptest! {
        #[test]
        fn no_nan_or_inf(inputs in arb_inputs()) {
            let mut bank = make_bank();
            apply_miner_input(&inputs, &mut bank);
            prop_assert!(bank.command.is_finite());
        }
    }

    #[test]
    fn roll_command_along_neg_z() {
        let mut bank = make_bank();
        let inputs: HashSet<PlayerInput> = [PlayerInput::RollCW].into_iter().collect();
        apply_miner_input(&inputs, &mut bank);
        let perp = bank.command - DVec3::NEG_Z * bank.command.dot(DVec3::NEG_Z);
        assert!(
            perp.length() < 1e-12,
            "command has off-axis component: {perp:?}"
        );
    }

    // ── apply_miner_translation ───────────────────────────────────────────

    #[test]
    fn no_thrust_input_linear_command_zero() {
        let mut bank = make_bank();
        apply_miner_translation(&HashSet::new(), &mut bank);
        assert_eq!(bank.linear_command, DVec3::ZERO);
    }

    #[test]
    fn opposite_thrust_keys_cancel() {
        for (a, b) in [
            (PlayerInput::ThrustUp, PlayerInput::ThrustDown),
            (PlayerInput::ThrustLeft, PlayerInput::ThrustRight),
            (PlayerInput::ThrustForward, PlayerInput::ThrustBackward),
        ] {
            let inputs: HashSet<PlayerInput> = [a, b].into_iter().collect();
            let mut bank = make_bank();
            apply_miner_translation(&inputs, &mut bank);
            assert_eq!(bank.linear_command, DVec3::ZERO);
        }
    }

    #[test]
    fn up_thrust_sets_positive_y_command() {
        let inputs: HashSet<PlayerInput> = [PlayerInput::ThrustUp].into_iter().collect();
        let mut bank = make_bank();
        apply_miner_translation(&inputs, &mut bank);
        assert!(
            bank.linear_command.y > 0.0,
            "W should set +Y body-space command"
        );
        assert!(bank.linear_command.x.abs() < 1e-12, "spurious X component");
        assert!(bank.linear_command.z.abs() < 1e-12, "spurious Z component");
    }

    #[test]
    fn right_thrust_sets_positive_x_command() {
        let inputs: HashSet<PlayerInput> = [PlayerInput::ThrustRight].into_iter().collect();
        let mut bank = make_bank();
        apply_miner_translation(&inputs, &mut bank);
        assert!(
            bank.linear_command.x > 0.0,
            "D should set +X body-space command"
        );
        assert!(bank.linear_command.y.abs() < 1e-12, "spurious Y component");
        assert!(bank.linear_command.z.abs() < 1e-12, "spurious Z component");
    }

    #[test]
    fn forward_thrust_sets_negative_z_command() {
        let inputs: HashSet<PlayerInput> = [PlayerInput::ThrustForward].into_iter().collect();
        let mut bank = make_bank();
        apply_miner_translation(&inputs, &mut bank);
        assert!(
            bank.linear_command.z < 0.0,
            "LShift should set -Z body-space command (body -Z = nose)"
        );
        assert!(bank.linear_command.x.abs() < 1e-12, "spurious X component");
        assert!(bank.linear_command.y.abs() < 1e-12, "spurious Y component");
    }

    // Integration: translation command → actual world-space vel via thruster system.
    #[test]
    fn up_thrust_produces_body_y_velocity() {
        let inputs: HashSet<PlayerInput> = [PlayerInput::ThrustUp].into_iter().collect();
        let mut body = make_body(); // identity orientation: body Y = world Y
        let mut bank = make_bank();
        apply_miner_translation(&inputs, &mut bank);
        apply_thrusters(&mut body, &mut bank, 1.0);
        assert!(body.vel.dot(DVec3::Y) > 0.0, "up thrust not along body +Y");
        let perp = body.vel - DVec3::Y * body.vel.dot(DVec3::Y);
        assert!(perp.length() < 1e-12, "spurious off-axis velocity");
    }

    #[test]
    fn translation_respects_orientation() {
        use std::f64::consts::FRAC_PI_2;
        // rotation_x(π/2): body +Y → world +Z
        let inputs: HashSet<PlayerInput> = [PlayerInput::ThrustUp].into_iter().collect();
        let mut body = make_body();
        body.orientation = DQuat::from_rotation_x(FRAC_PI_2);
        let mut bank = make_bank();
        apply_miner_translation(&inputs, &mut bank);
        apply_thrusters(&mut body, &mut bank, 1.0);
        assert!(body.vel.length() > 1e-9, "no velocity after thrust");
        assert!(body.vel.z > 0.0, "body +Y should map to world +Z");
    }

    // ── apply_linear_damping ──────────────────────────────────────────────

    fn damping_inputs() -> HashSet<PlayerInput> {
        [PlayerInput::Damping].into_iter().collect()
    }

    #[test]
    fn damping_inactive_without_tab() {
        let mut body = make_body();
        body.vel = DVec3::new(3.0, 0.0, 0.0);
        let mut bank = make_bank();
        apply_linear_damping(&HashSet::new(), &body, &mut bank);
        assert_eq!(
            bank.linear_command,
            DVec3::ZERO,
            "no TAB → no damping command"
        );
    }

    #[test]
    fn damping_opposes_velocity() {
        let mut body = make_body();
        body.vel = DVec3::new(3.0, 0.0, 0.0);
        let mut bank = make_bank();
        apply_linear_damping(&damping_inputs(), &body, &mut bank);
        assert!(
            bank.linear_command.x < 0.0,
            "retro command should oppose +X velocity"
        );
    }

    #[test]
    fn damping_command_opposes_vel_direction() {
        let dir = DVec3::new(1.0, 2.0, 3.0).normalize();
        let mut body = make_body();
        body.vel = dir * 5.0;
        let mut bank = make_bank();
        apply_linear_damping(&damping_inputs(), &body, &mut bank);
        assert!(
            bank.linear_command.dot(dir) < 0.0,
            "damping command must oppose velocity direction"
        );
    }

    // ── apply_angular_damping ─────────────────────────────────────────────

    #[test]
    fn angular_damping_inactive_without_tab() {
        let mut body = make_body();
        body.angular_vel = DVec3::new(0.0, 0.0, 3.0);
        let mut bank = make_bank();
        apply_angular_damping(&HashSet::new(), &body, &mut bank);
        assert_eq!(
            bank.command,
            DVec3::ZERO,
            "no TAB → no angular damping command"
        );
    }

    #[test]
    fn angular_damping_opposes_roll() {
        // Pure roll around body -Z (ship_fwd with identity orientation = -Z).
        let mut body = make_body();
        body.angular_vel = DVec3::NEG_Z * 3.0; // rolling CW
        let mut bank = make_bank();
        apply_angular_damping(&damping_inputs(), &body, &mut bank);
        // Command should oppose roll: command is in +Z, so dot with NEG_Z is negative.
        assert!(
            bank.command.dot(DVec3::NEG_Z) < 0.0,
            "command must oppose -Z roll; command={:?}",
            bank.command
        );
    }

    #[test]
    fn angular_damping_ignores_heading_spin() {
        // Pure yaw (around world Y) with identity orientation.
        let mut body = make_body();
        body.angular_vel = DVec3::Y * 3.0;
        let mut bank = make_bank();
        apply_angular_damping(&damping_inputs(), &body, &mut bank);
        // Y spin has zero roll component → no command written
        assert!(
            bank.command.length() < 1e-12,
            "heading spin must not produce damping command"
        );
    }

    // Integration: TAB held should reduce velocity over time.
    #[test]
    fn damping_decays_velocity_over_time() {
        let mut body = make_body();
        body.vel = DVec3::new(2.0, 0.0, 0.0);
        let dt = 1.0 / 60.0;
        for _ in 0..30 {
            let mut bank = make_bank();
            apply_linear_damping(&damping_inputs(), &body, &mut bank);
            apply_thrusters(&mut body, &mut bank, dt);
        }
        assert!(
            body.vel.length() < 2.0,
            "velocity should decrease: |vel|={:.3}",
            body.vel.length()
        );
        assert!(
            body.vel.length() > 0.0,
            "damping should not zero vel instantly"
        );
    }
}
