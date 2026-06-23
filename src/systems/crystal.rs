use legion::{system, systems::CommandBuffer, world::SubWorld, *};
use roxlap_gpu::GpuRenderer;

use crate::{
    components::{
        crystal::CrystalMarker, miner::Miner, newton_body::NewtonBody, sprite_id::Sprite,
    },
    generation::chunks::{world_to_chunk, LOAD_RADIUS},
    systems::presence_position::{build_sprite_maps, perform_despawn},
};

/// Ship-centre to crystal-centre distance at which the crystal is consumed.
const CRYSTAL_PICKUP_RADIUS_SQ: f64 = 3.0 * 3.0;

#[system]
#[read_component(Miner)]
#[read_component(CrystalMarker)]
#[read_component(NewtonBody)]
#[write_component(Sprite)]
pub fn crystal(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] gpu: &mut GpuRenderer,
) {
    let (ship_pos, ship_chunk) = {
        let mut q = <(&Miner, &NewtonBody)>::query();
        let (_, body) = q.iter(world).next().expect("miner missing");
        (body.pos, world_to_chunk(body.pos))
    };

    let r2 = LOAD_RADIUS * LOAD_RADIUS;

    let mut to_despawn: Vec<Entity> = Vec::new();
    {
        let mut q = <(Entity, &CrystalMarker, &NewtonBody)>::query();
        for (entity, _, body) in q.iter(world) {
            let picked_up = (body.pos - ship_pos).length_squared() <= CRYSTAL_PICKUP_RADIUS_SQ;
            let dc = world_to_chunk(body.pos) - ship_chunk;
            let out_of_range = dc.dot(dc) > r2;
            if picked_up || out_of_range {
                to_despawn.push(*entity);
            }
        }
    }

    if to_despawn.is_empty() {
        return;
    }

    let mut maps = build_sprite_maps(world);
    for entity in to_despawn {
        perform_despawn(entity, &mut maps, world, commands, gpu);
    }
}
