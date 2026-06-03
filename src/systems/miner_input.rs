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

        // Track which world-space axes are actively driven this frame so
        // deceleration targets the exact same axes as acceleration.
        let mut pitch_axis: Option<DVec3> = None;
        let mut yaw_axis: Option<DVec3> = None;
        let mut roll_axis: Option<DVec3> = None;
        let mut thrust_driven = false;

        for input in inputs {
            match input {
                PlayerInput::PitchCW => {
                    body.angular_vel += right * ANGULAR_ACCEL * dt.0;
                    pitch_axis = Some(right);
                }
                PlayerInput::PitchCCW => {
                    body.angular_vel -= right * ANGULAR_ACCEL * dt.0;
                    pitch_axis = Some(right);
                }
                PlayerInput::YawCW => {
                    body.angular_vel += up * ANGULAR_ACCEL * dt.0;
                    yaw_axis = Some(up);
                }
                PlayerInput::YawCCW => {
                    body.angular_vel -= up * ANGULAR_ACCEL * dt.0;
                    yaw_axis = Some(up);
                }
                PlayerInput::RollCW => {
                    body.angular_vel += forward * ANGULAR_ACCEL * dt.0;
                    roll_axis = Some(forward);
                }
                PlayerInput::RollCCW => {
                    body.angular_vel -= forward * ANGULAR_ACCEL * dt.0;
                    roll_axis = Some(forward);
                }
                PlayerInput::IncTrust => {
                    body.vel += forward * LINEAR_ACCEL * dt.0;
                    thrust_driven = true;
                }
                PlayerInput::DecTrust => {
                    body.vel -= forward * LINEAR_ACCEL * dt.0;
                    thrust_driven = true;
                }
            }
        }

        // Decelerate unpressed axes using the same world-space axis that
        // acceleration used, so damp and drive act on exactly the same component.
        if pitch_axis.is_none() {
            damp_axis(&mut body.angular_vel, right, damp);
        }
        if yaw_axis.is_none() {
            damp_axis(&mut body.angular_vel, up, damp);
        }
        if roll_axis.is_none() {
            damp_axis(&mut body.angular_vel, forward, damp);
        }
        if !thrust_driven {
            damp_axis(&mut body.vel, forward, LINEAR_ACCEL * dt.0);
        }
    }
}
