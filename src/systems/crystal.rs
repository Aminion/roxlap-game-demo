use legion::{system, systems::CommandBuffer, world::SubWorld, *};
use roxlap_gpu::GpuRenderer;

use crate::{
    components::{
        crystal::CrystalMarker, miner::Miner, newton_body::NewtonBody, sprite_id::Sprite,
    },
    generation::chunks::{world_to_chunk, LOAD_RADIUS},
    systems::{
        energy::{Energy, ENERGY_MAX},
        sprite::{build_sprite_maps, perform_despawn},
    },
    Dt,
};

const CRYSTAL_PICKUP_RADIUS_SQ: f64 = 3.0 * 3.0;
const CRYSTAL_REGEN_DIST_SQ: f64 = 8.0 * 8.0;
pub const CRYSTAL_REGEN_RATE: f64 = 25.0;

fn compute_regen(current: f64, near_count: usize, dt: f64) -> f64 {
    if near_count == 0 {
        return current;
    }
    (current + CRYSTAL_REGEN_RATE * near_count as f64 * dt).min(ENERGY_MAX)
}

#[system]
#[read_component(Miner)]
#[read_component(CrystalMarker)]
#[read_component(NewtonBody)]
#[write_component(Sprite)]
pub fn crystal(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] gpu: &mut GpuRenderer,
    #[resource] energy: &mut Energy,
    #[resource] dt: &Dt,
) {
    let (ship_pos, ship_chunk) = {
        let mut q = <(&Miner, &NewtonBody)>::query();
        let (_, body) = q.iter(world).next().expect("miner missing");
        (body.pos, world_to_chunk(body.pos))
    };

    let r2 = LOAD_RADIUS * LOAD_RADIUS;

    let mut to_despawn: Vec<Entity> = Vec::new();
    let mut near_count = 0usize;
    {
        let mut q = <(Entity, &CrystalMarker, &NewtonBody)>::query();
        for (entity, _, body) in q.iter(world) {
            let dist_sq = (body.pos - ship_pos).length_squared();
            let picked_up = dist_sq <= CRYSTAL_PICKUP_RADIUS_SQ;
            let dc = world_to_chunk(body.pos) - ship_chunk;
            let out_of_range = dc.dot(dc) > r2;
            if picked_up || out_of_range {
                to_despawn.push(*entity);
            } else if dist_sq <= CRYSTAL_REGEN_DIST_SQ {
                near_count += 1;
            }
        }
    }

    energy.current = compute_regen(energy.current, near_count, dt.0);

    if to_despawn.is_empty() {
        return;
    }

    let mut maps = build_sprite_maps(world);
    for entity in to_despawn {
        perform_despawn(entity, &mut maps, world, commands, gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_regen_without_crystals() {
        assert_eq!(compute_regen(50.0, 0, 1.0), 50.0);
    }

    #[test]
    fn single_crystal_adds_correct_amount() {
        let result = compute_regen(0.0, 1, 1.0);
        assert!((result - CRYSTAL_REGEN_RATE).abs() < 1e-12);
    }

    #[test]
    fn two_crystals_add_double() {
        let result = compute_regen(0.0, 2, 1.0);
        assert!((result - 2.0 * CRYSTAL_REGEN_RATE).abs() < 1e-12);
    }

    #[test]
    fn regen_caps_at_energy_max() {
        let result = compute_regen(ENERGY_MAX - 1.0, 1, 1.0);
        assert_eq!(result, ENERGY_MAX);
    }
}
