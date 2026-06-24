use glam::DVec3;
use legion::{world::SubWorld, *};

use crate::{
    components::{aabb::Aabb, crystal::CrystalMarker, miner::Miner, newton_body::NewtonBody},
    math::ray_aabb,
    systems::energy::Energy,
    Dt, Retrieving,
};

const RETRIEVAL_ACCEL: f64 = 30.0;
const RETRIEVAL_ENERGY_DRAIN: f64 = 5.0;


#[system]
#[read_component(Miner)]
#[read_component(CrystalMarker)]
#[read_component(Aabb)]
#[write_component(NewtonBody)]
pub fn retrieval(
    world: &mut SubWorld,
    #[resource] retrieving: &Retrieving,
    #[resource] energy: &mut Energy,
    #[resource] dt: &Dt,
) {
    if !retrieving.0 {
        return;
    }

    let dt = dt.0;
    let cost = RETRIEVAL_ENERGY_DRAIN * dt;
    if energy.current < cost {
        return;
    }
    energy.current -= cost;

    let (miner_pos, forward) = {
        let mut q = <(&Miner, &NewtonBody)>::query();
        let (_, body) = q.iter(world).next().expect("miner missing");
        (body.pos, (body.orientation * DVec3::NEG_Z).normalize())
    };

    // Find the closest crystal whose AABB intersects the ship's forward ray.
    // Collect (entity, crystal_pos) pairs so we can apply the impulse after.
    let target: Option<(Entity, DVec3)> = {
        let mut q = <(Entity, &CrystalMarker, &NewtonBody, &Aabb)>::query();
        q.iter(world)
            .filter_map(|(entity, _, body, aabb)| {
                let t = ray_aabb(miner_pos, forward, body.pos, aabb.half_extent as f64)?;
                Some((*entity, body.pos, t))
            })
            .min_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(e, pos, _)| (e, pos))
    };

    let Some((target_entity, crystal_pos)) = target else {
        return;
    };

    let to_ship = miner_pos - crystal_pos;
    if let Some(dir) = to_ship.try_normalize() {
        if let Ok(mut entry) = world.entry_mut(target_entity) {
            if let Ok(body) = entry.get_component_mut::<NewtonBody>() {
                body.vel += dir * RETRIEVAL_ACCEL * dt;
            }
        }
    }
}
