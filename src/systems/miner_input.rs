use std::collections::HashSet;

use glam::DVec3;
use legion::{world::SubWorld, *};

use crate::{
    components::{miner::Miner, newton_body::NewtonBody},
    Dt, PlayerInput,
};

const ANGULAR_ACCEL: f64 = 1.2;
const LINEAR_ACCEL: f64 = 20.0;

/// Damp `vel`'s component along `axis` toward zero by at most `amount`,
/// without overshooting.
fn damp_axis(vel: &mut DVec3, axis: DVec3, amount: f64) {
    let v = vel.dot(axis);
    *vel -= axis * v.signum() * amount.min(v.abs());
}

#[system]
#[read_component(Miner)]
#[write_component(NewtonBody)]
pub fn miner_input(
    world: &mut SubWorld,
    #[resource] inputs: &HashSet<PlayerInput>,
    #[resource] dt: &Dt,
) {
    let mut query = <(&Miner, &mut NewtonBody)>::query();
    for (_, body) in query.iter_mut(world) {
        let forward = body.orientation * DVec3::NEG_Z;
        let right = body.orientation * DVec3::X;
        let up = body.orientation * DVec3::Y;

        let damp = ANGULAR_ACCEL * dt.0;

        // Accumulate net input per axis. Opposite keys cancel to zero,
        // which correctly triggers damping instead of freezing velocity.
        let mut net_pitch: f64 = 0.0;
        let mut net_yaw: f64 = 0.0;
        let mut net_roll: f64 = 0.0;
        let mut net_thrust: f64 = 0.0;

        for input in inputs {
            match input {
                PlayerInput::PitchCW => net_pitch += 1.0,
                PlayerInput::PitchCCW => net_pitch -= 1.0,
                PlayerInput::YawCW => net_yaw += 1.0,
                PlayerInput::YawCCW => net_yaw -= 1.0,
                PlayerInput::RollCW => net_roll += 1.0,
                PlayerInput::RollCCW => net_roll -= 1.0,
                PlayerInput::IncTrust => net_thrust += 1.0,
                PlayerInput::DecTrust => net_thrust -= 1.0,
            }
        }

        if net_pitch != 0.0 {
            body.angular_vel += right * ANGULAR_ACCEL * dt.0 * net_pitch;
        } else {
            damp_axis(&mut body.angular_vel, right, damp);
        }
        if net_yaw != 0.0 {
            body.angular_vel += up * ANGULAR_ACCEL * dt.0 * net_yaw;
        } else {
            damp_axis(&mut body.angular_vel, up, damp);
        }
        if net_roll != 0.0 {
            body.angular_vel += forward * ANGULAR_ACCEL * dt.0 * net_roll;
        } else {
            damp_axis(&mut body.angular_vel, forward, damp);
        }
        if net_thrust != 0.0 {
            body.vel += forward * LINEAR_ACCEL * dt.0 * net_thrust;
        } else {
            damp_axis(&mut body.vel, forward, LINEAR_ACCEL * dt.0);
        }
    }
}
