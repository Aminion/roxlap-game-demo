use glam::DVec3;
use legion::{world::SubWorld, *};

use crate::components::{camera::CameraComponent, miner::Miner, newton_body::NewtonBody};

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
    cam.0.position = body.pos.as_vec3().to_array();
    cam.0.forward = fwd.as_vec3().to_array();
    cam.0.right = right.as_vec3().to_array();
    cam.0.down = (-up).as_vec3().to_array();
}
