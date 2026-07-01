use glam::Vec3;
use legion::{system, world::SubWorld, *};
use roxlap_render::DirectionalLight;

use crate::components::{camera::CameraComponent, miner::Miner, newton_body::NewtonBody};

const HEADLIGHT_INTENSITY: f32 = 0.25;

pub struct Headlight(pub Option<DirectionalLight>);

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[read_component(CameraComponent)]
pub fn lighting(world: &SubWorld, #[resource] headlight: &mut Headlight) {
    let miner_pos = {
        let mut q = <(&Miner, &NewtonBody)>::query();
        q.iter(world).next().map(|(_, body)| body.pos.as_vec3())
    };
    let cam_pos = {
        let mut q = <&CameraComponent>::query();
        q.iter(world)
            .next()
            .map(|cam| Vec3::from_array(cam.0.pos.map(|v| v as f32)))
    };
    headlight.0 = miner_pos.zip(cam_pos).map(|(mp, cam)| {
        let dir = (mp - cam).normalize_or_zero();
        DirectionalLight {
            direction: dir.to_array(),
            color: [1.0; 3],
            intensity: HEADLIGHT_INTENSITY,
            casts_shadow: true,
        }
    });
}
