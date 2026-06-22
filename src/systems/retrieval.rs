use glam::DVec3;
use legion::{world::SubWorld, *};

use crate::{
    components::{aabb::Aabb, asteroid::CrystalMarker, miner::Miner, newton_body::NewtonBody},
    systems::energy::Energy,
    Dt, Retrieving,
};

const RETRIEVAL_ACCEL: f64 = 30.0;
const RETRIEVAL_ENERGY_DRAIN: f64 = 5.0;

/// Slab-method ray–AABB test. Returns the entry t along `ray_dir`, or `None`.
fn ray_aabb(ray_origin: DVec3, ray_dir: DVec3, center: DVec3, half: f64) -> Option<f64> {
    let mut t_min = f64::NEG_INFINITY;
    let mut t_max = f64::INFINITY;
    for i in 0..3 {
        let o = ray_origin[i];
        let d = ray_dir[i];
        let c = center[i];
        if d.abs() < 1e-12 {
            if (o - c).abs() > half {
                return None;
            }
        } else {
            let (t1, t2) = {
                let a = (c - half - o) / d;
                let b = (c + half - o) / d;
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            };
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
        }
    }
    if t_max < t_min || t_max < 0.0 {
        return None;
    }
    Some(if t_min >= 0.0 { t_min } else { t_max })
}

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
        let Some((_, body)) = q.iter(world).next() else {
            return;
        };
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
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
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
