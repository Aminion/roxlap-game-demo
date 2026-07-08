use legion::{system, systems::CommandBuffer, world::SubWorld, *};
use roxlap_render::SceneRenderer;

use crate::{
    components::{
        aabb::Aabb, crystal::CrystalMarker, energy::Energy, miner::Miner, newton_body::NewtonBody,
        sprite_id::Sprite,
    },
    generation::chunks::{world_to_chunk, LOAD_RADIUS},
    systems::sprite::perform_despawn,
};

const CRYSTAL_PICKUP_ENERGY: f64 = 25.0;

#[system]
#[read_component(Miner)]
#[read_component(CrystalMarker)]
#[read_component(NewtonBody)]
#[read_component(Aabb)]
#[write_component(Sprite)]
pub fn crystal(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] renderer: &mut SceneRenderer,
    #[resource] energy: &mut Energy,
) {
    let (ship_chunk, ship_aabb) = {
        let mut q = <(&Miner, &NewtonBody, &Aabb)>::query();
        let (_, body, aabb) = q.iter(world).next().expect("miner missing");
        (world_to_chunk(body.pos), aabb.clone())
    };

    let r2 = LOAD_RADIUS * LOAD_RADIUS;

    let mut to_despawn: Vec<Entity> = Vec::new();
    let mut pickup_count = 0usize;
    {
        let mut q = <(Entity, &CrystalMarker, &NewtonBody, &Aabb)>::query();
        for (entity, _, body, crystal_aabb) in q.iter(world) {
            let picked_up = ship_aabb.overlaps(crystal_aabb);
            let dc = world_to_chunk(body.pos) - ship_chunk;
            let out_of_range = dc.dot(dc) > r2;
            if picked_up {
                pickup_count += 1;
                to_despawn.push(*entity);
            } else if out_of_range {
                to_despawn.push(*entity);
            }
        }
    }

    if pickup_count > 0 {
        energy.current = energy.current + CRYSTAL_PICKUP_ENERGY * pickup_count as f64;
    }

    for entity in to_despawn {
        perform_despawn(entity, world, commands, renderer);
    }
}

#[cfg(test)]
mod tests {}
