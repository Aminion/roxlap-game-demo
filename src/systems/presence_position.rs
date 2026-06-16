use bytemuck::Zeroable;
use glam::{DQuat, DVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, *};
use rand::RngExt;
use roxlap_gpu::{GpuRenderer, SpriteInstance, SpriteInstanceTransform};

use crate::{
    components::{
        asteroid::AsteroidMarker, miner::Miner, newton_body::NewtonBody,
        presence_position::PresencePosition, sprite_id::SpriteId,
    },
    generation::chunks::{missing_chunks, CHUNK_SIZE, LOAD_RADIUS},
    world::build_asteroid_sprite_model,
    GeneratedChunks, SpriteData,
};

const ASTEROIDS_PER_CHUNK: u32 = 1;
const UPDATE_DIST_SQ: f64 = (CHUNK_SIZE as f64 / 2.0) * (CHUNK_SIZE as f64 / 2.0);

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[write_component(PresencePosition)]
pub fn presence_position_update(
    #[resource] generated: &mut GeneratedChunks,
    #[resource] gpu: &mut GpuRenderer,
    #[resource] sprite_data: &mut SpriteData,
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
) {
    let mut updated_pos: Option<DVec3> = None;
    {
        let mut query = <(&Miner, &NewtonBody, &mut PresencePosition)>::query();
        for (_, body, presence) in query.iter_mut(world) {
            if body.pos.distance_squared(presence.0) > UPDATE_DIST_SQ {
                presence.0 = body.pos;
                updated_pos = Some(body.pos);
            }
        }
    }

    let ship_pos = match updated_pos {
        Some(p) => p,
        None => return,
    };

    let to_generate: Vec<_> = missing_chunks(ship_pos, LOAD_RADIUS, &generated.0);

    if to_generate.is_empty() {
        return;
    }

    let mut rng = rand::rng();
    let placeholder = SpriteInstanceTransform::zeroed();

    for &chunk in &to_generate {
        let chunk_centre = (chunk.as_dvec3() + DVec3::splat(0.5)) * CHUNK_SIZE as f64;
        for _ in 0..ASTEROIDS_PER_CHUNK {
            let chain_id = sprite_data.registry.add(build_asteroid_sprite_model());
            gpu.add_sprite_model(&sprite_data.registry, chain_id);
            let slot = gpu.append_sprite_instances(
                &sprite_data.registry,
                &[SpriteInstance {
                    model_id: chain_id,
                    transform: placeholder,
                }],
            );
            let angular_vel = DVec3::new(
                (rng.random::<f64>() - 0.5) * 2.0,
                (rng.random::<f64>() - 0.5) * 2.0,
                (rng.random::<f64>() - 0.5) * 2.0,
            );
            commands.push((
                AsteroidMarker,
                SpriteId { model_id: slot },
                NewtonBody {
                    mass: 1.0,
                    pos: chunk_centre,
                    vel: DVec3::ZERO,
                    orientation: DQuat::IDENTITY,
                    angular_vel,
                },
            ));
        }
        generated.0.insert(chunk);
    }
}
