use glam::Vec3;
use legion::{system, world::SubWorld, *};
use roxlap_render::DirectionalLight;

use crate::components::{
    camera::CameraComponent, headlight::Headlight, miner::Miner, newton_body::NewtonBody,
};

const HEADLIGHT_INTENSITY: f32 = 0.25;

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[read_component(CameraComponent)]
#[write_component(Headlight)]
pub fn lighting(world: &mut SubWorld) {
    let mut q = <(&Miner, &NewtonBody, &CameraComponent, &mut Headlight)>::query();
    for (_, body, cam, headlight) in q.iter_mut(world) {
        let mp = body.pos.as_vec3();
        let cam_pos = Vec3::from_array(cam.0.pos.map(|v| v as f32));
        let dir = (mp - cam_pos).normalize_or_zero();
        headlight.0 = Some(DirectionalLight {
            direction: dir.to_array(),
            color: [1.0; 3],
            intensity: HEADLIGHT_INTENSITY,
            casts_shadow: true,
        });
    }
}
