use glam::Vec3;
use legion::{system, world::SubWorld, *};
use roxlap_render::{DirectionalLight, PointLight};

use crate::components::{
    camera::CameraComponent, crystal::CrystalMarker, headlight::Headlight, miner::Miner,
    newton_body::NewtonBody,
};

const HEADLIGHT_INTENSITY: f32 = 0.25;

pub struct PointLights(pub Vec<PointLight>);

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[read_component(CameraComponent)]
#[write_component(Headlight)]
#[read_component(CrystalMarker)]
pub fn lighting(world: &mut SubWorld, #[resource] point_lights: &mut PointLights) {
    {
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

    point_lights.0.clear();
    let mut q = <(&CrystalMarker, &NewtonBody)>::query();
    for (_, body) in q.iter(world) {
        point_lights.0.push(PointLight {
            position: body.pos.as_vec3().to_array(),
            color: [1.0, 0.0, 0.0],
            intensity: 1.5,
            radius: 24.0,
            casts_shadow: false,
        });
    }
}
