use bytemuck::Zeroable;
use glam::{DQuat, DVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, IntoQuery};
use rand::RngExt;
use roxlap_gpu::{GpuRenderer, SpriteInstance, SpriteInstanceTransform};

use crate::{
    components::{asteroid::AsteroidMarker, miner::Miner, newton_body::NewtonBody},
    generation::chunks::{missing_chunks, CHUNK_SIZE, LOAD_RADIUS},
    systems::render::sprite_from_body,
    world::build_asteroid_sprite_model,
    GeneratedChunks, SpriteData,
};

/// Identity-orientation instance transform at `pos` — the spawn pose of a
/// fresh asteroid (zero velocity, `DQuat::IDENTITY`). Built explicitly so the
/// uploaded instance buffer never contains a degenerate (all-zero `inv_rot`)
/// transform, which the sprite marcher would otherwise rasterise as a large
/// solid quad at the world origin.
fn identity_instance(model_id: u32, pos: DVec3) -> SpriteInstance {
    let mut transform = SpriteInstanceTransform::zeroed();
    transform.inv_rot = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    transform.pos = pos.as_vec3().to_array();
    SpriteInstance {
        model_id,
        transform,
    }
}

/// Asteroids spawned per chunk (placed at chunk centre).
const ASTEROIDS_PER_CHUNK: u32 = 1;
/// Maximum chunks populated per tick to avoid first-frame hitching.
const CHUNKS_PER_TICK: usize = 32;

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[read_component(AsteroidMarker)]
pub fn chunk_population(
    #[resource] generated: &mut GeneratedChunks,
    #[resource] gpu: &mut GpuRenderer,
    #[resource] sprite_data: &mut SpriteData,
    world: &SubWorld,
    commands: &mut CommandBuffer,
) {
    let ship_pos = {
        let mut q = <(&Miner, &NewtonBody)>::query();
        match q.iter(world).next() {
            Some((_, body)) => body.pos,
            None => return,
        }
    };

    let to_generate: Vec<_> = missing_chunks(ship_pos, LOAD_RADIUS, &generated.0)
        .into_iter()
        .take(CHUNKS_PER_TICK)
        .collect();

    if to_generate.is_empty() {
        return;
    }

    let mut rng = rand::rng();
    let mut next_id = sprite_data.instance_count;

    // Spawn poses of the asteroids created this tick, by model id. The
    // entities themselves are deferred through `commands` and don't exist in
    // `world` yet (the schedule flushes the command buffer after this system),
    // so we record their poses here to seed the instance buffer honestly.
    let mut spawned: Vec<(u32, DVec3)> = Vec::new();

    for &chunk in &to_generate {
        let chunk_centre = (chunk.as_dvec3() + DVec3::splat(0.5)) * CHUNK_SIZE as f64;
        for _ in 0..ASTEROIDS_PER_CHUNK {
            // Each asteroid gets its own model so individual voxels can be
            // edited independently when the asteroid is damaged or destroyed.
            sprite_data.registry.add(build_asteroid_sprite_model());
            let angular_vel = DVec3::new(
                (rng.random::<f64>() - 0.5) * 2.0,
                (rng.random::<f64>() - 0.5) * 2.0,
                (rng.random::<f64>() - 0.5) * 2.0,
            );
            commands.push((
                AsteroidMarker { model_id: next_id },
                NewtonBody {
                    mass: 1.0,
                    pos: chunk_centre,
                    vel: DVec3::ZERO,
                    orientation: DQuat::IDENTITY,
                    angular_vel,
                },
            ));
            spawned.push((next_id, chunk_centre));
            next_id += 1;
        }
        generated.0.insert(chunk);
    }

    // Rebuild the full instance list (slot i uses model i, 1:1). Seed every
    // slot with a valid identity transform so the upload is never degenerate,
    // then fill real poses: already-live asteroids from their `NewtonBody`,
    // and the just-spawned ones from their recorded spawn centre. Render
    // re-poses these again the same frame, but keeping the upload honest means
    // a transient or mis-ordered frame can't flash a quad at the origin.
    let mut instances: Vec<SpriteInstance> = (0..next_id)
        .map(|id| identity_instance(id, DVec3::ZERO))
        .collect();

    let mut q = <(&AsteroidMarker, &NewtonBody)>::query();
    for (marker, body) in q.iter(world) {
        if let Some(slot) = instances.get_mut(marker.model_id as usize) {
            *slot = sprite_from_body(body, marker.model_id);
        }
    }
    for &(id, centre) in &spawned {
        instances[id as usize] = identity_instance(id, centre);
    }

    gpu.set_sprite_instances(&sprite_data.registry, &instances);
    sprite_data.instance_count = next_id;
}
