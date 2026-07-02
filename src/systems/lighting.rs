use glam::DVec3;
use legion::{system, world::SubWorld, *};
use roxlap_render::{PointLight, SpotLight};

use crate::{
    components::{crystal::CrystalMarker, miner::Miner, newton_body::NewtonBody},
    world::MinerModel,
};

pub struct PointLights(pub Vec<PointLight>);
pub struct SpotLights(pub Vec<SpotLight>);

const NOSE_SPOT_INNER_DEG: f32 = 20.0;
const NOSE_SPOT_OUTER_DEG: f32 = 40.0;
const NOSE_SPOT_RADIUS: f32 = 80.0;
const NOSE_SPOT_INTENSITY: f32 = 1.5;

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[read_component(CrystalMarker)]
pub fn lighting(
    world: &mut SubWorld,
    #[resource] point_lights: &mut PointLights,
    #[resource] spot_lights: &mut SpotLights,
    #[resource] miner_model: &MinerModel,
) {
    point_lights.0.clear();
    spot_lights.0.clear();

    let mut q = <(&Miner, &NewtonBody)>::query();
    for (_, body) in q.iter(world) {
        point_lights.0.push(PointLight {
            position: body.pos.as_vec3().to_array(),
            color: [1.0; 3],
            intensity: 1.0,
            radius: miner_model.radius,
            casts_shadow: true,
        });

        let fwd = (body.orientation * DVec3::NEG_Z).as_vec3();
        let nose_pos = body.pos.as_vec3() + fwd * miner_model.nose_offset as f32;
        spot_lights.0.push(SpotLight {
            position: nose_pos.to_array(),
            direction: fwd.to_array(),
            color: [0.75, 0.88, 1.0],
            intensity: NOSE_SPOT_INTENSITY,
            radius: NOSE_SPOT_RADIUS,
            inner_angle_deg: NOSE_SPOT_INNER_DEG,
            outer_angle_deg: NOSE_SPOT_OUTER_DEG,
            casts_shadow: true,
        });
    }

    let mut q = <(&CrystalMarker, &NewtonBody)>::query();
    for (_, body) in q.iter(world) {
        point_lights.0.push(PointLight {
            position: body.pos.as_vec3().to_array(),
            color: [1.0, 0.0, 0.0],
            intensity: 1.5,
            radius: 24.0,
            casts_shadow: true,
        });
    }
}
