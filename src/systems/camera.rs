use glam::DVec3;
use legion::{world::SubWorld, *};

use crate::components::{camera::CameraComponent, miner::Miner, newton_body::NewtonBody};

const THIRD_PERSON_DIST: f64 = 48.0;
const THIRD_PERSON_HEIGHT: f64 = 16.0;

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[write_component(CameraComponent)]
pub fn camera_update(world: &mut SubWorld) {
    let mut query = <(&Miner, &NewtonBody, &mut CameraComponent)>::query();
    let (_, body, cam) = query.iter_mut(world).next().expect("miner missing");
    let fwd = body.orientation * DVec3::NEG_Z;
    let right = body.orientation * DVec3::X;
    let up = body.orientation * DVec3::Y;
    let cam_pos = body.pos - fwd * THIRD_PERSON_DIST + up * THIRD_PERSON_HEIGHT;
    cam.0.pos = cam_pos.to_array();
    cam.0.forward = fwd.to_array();
    cam.0.right = right.to_array();
    cam.0.down = (-up).to_array();
}
