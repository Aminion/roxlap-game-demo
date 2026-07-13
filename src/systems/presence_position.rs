use glam::{DQuat, DVec3, IVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, *};
use rayon::prelude::*;
use roxlap_cavegen::PerlinNoise3D;
use roxlap_gpu::SpriteModelRegistry;
use roxlap_render::{Kv6, SceneRenderer};

use crate::{
    components::{
        aabb::Aabb,
        asteroid::{AsteroidMarker, AsteroidMinerals, AsteroidVoxelInfo},
        miner::Miner,
        newton_body::NewtonBody,
        presence_position::PresencePosition,
        sprite_id::Sprite,
    },
    generation::chunks::{
        chunk_has_asteroid, compute_spawn, missing_chunks, world_to_chunk, ChunkQueue,
        LoadedAsteroids, SpawnData, VisitedChunks, WorldSeed, CHUNK_SIZE, LOAD_RADIUS,
        UNLOAD_RADIUS,
    },
    systems::sprite::{perform_despawn, spawn_sprite, sprite_model_to_kv6},
};

const UPDATE_DIST_SQ: f64 = (CHUNK_SIZE as f64 / 2.0) * (CHUNK_SIZE as f64 / 2.0);

/// Asteroid spawns allowed per frame. The expensive per-spawn work (model
/// build ~0.8ms + kv6 conversion ~0.3ms, measured in release) runs in the
/// parallel phase; only registry/renderer registration stays on the main
/// thread, so this cap is bounded by the parallel batch's wall time
/// (~32 × 1.1ms ÷ cores). A fresh load sphere holds ~1160 asteroids
/// (seed 42), so 32/frame repopulates it in ~0.6s at 60 fps.
const MAX_SPAWNS_PER_FRAME: usize = 32;

/// Chunk evaluations (density noise) allowed per frame. The gate costs
/// ~0.14µs per chunk, so this bounds queue-scan bookkeeping more than CPU:
/// a fresh load sphere (~2100 chunks) is fully triaged within ~8 frames.
const MAX_CHUNK_EVALS_PER_FRAME: usize = 256;

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[read_component(Sprite)]
#[write_component(PresencePosition)]
pub fn presence_position_update(
    #[resource] visited: &mut VisitedChunks,
    #[resource] loaded: &mut LoadedAsteroids,
    #[resource] renderer: &mut SceneRenderer,
    #[resource] sprite_data: &mut SpriteModelRegistry,
    #[resource] world_seed: &WorldSeed,
    #[resource] chunk_queue: &mut ChunkQueue,
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
) {
    let (ship_pos, updated_pos) = {
        let mut query = <(&Miner, &NewtonBody, &mut PresencePosition)>::query();
        let (_, body, presence) = query.iter_mut(world).next().expect("miner missing");
        let pos = body.pos;
        let updated = if pos.distance_squared(presence.0) > UPDATE_DIST_SQ {
            presence.0 = pos;
            true
        } else {
            false
        };
        (pos, updated)
    };

    if updated_pos {
        unload_departed_chunks(
            ship_pos,
            visited,
            loaded,
            renderer,
            sprite_data,
            world,
            commands,
        );
        for chunk in missing_chunks(ship_pos, LOAD_RADIUS, &visited.0) {
            // enqueue() is a no-op if the chunk is already in the queue set.
            chunk_queue.enqueue(chunk);
        }
    }

    spawn_queued_chunks(
        ship_pos,
        chunk_queue,
        visited,
        loaded,
        renderer,
        sprite_data,
        commands,
        world_seed.0,
    );
}

/// Despawn asteroids whose *spawn* chunk left the unload radius and forget all
/// evaluated chunks outside it. Keying on the spawn chunk (not the asteroid's
/// current position — mining impulses push asteroids across chunk borders)
/// keeps despawn and re-population in lockstep: a chunk that re-enters the
/// load radius regenerates the same asteroid deterministically.
fn unload_departed_chunks(
    ship_pos: DVec3,
    visited: &mut VisitedChunks,
    loaded: &mut LoadedAsteroids,
    renderer: &mut SceneRenderer,
    registry: &mut SpriteModelRegistry,
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
) {
    let center = world_to_chunk(ship_pos);
    let r2 = UNLOAD_RADIUS * UNLOAD_RADIUS;

    let to_unload: Vec<Entity> = loaded
        .0
        .iter()
        .filter(|(_, &chunk)| (chunk - center).length_squared() > r2)
        .map(|(&entity, _)| entity)
        .collect();
    for entity in to_unload {
        perform_despawn(entity, world, commands, renderer, registry);
        loaded.0.remove(&entity);
    }

    visited.0.retain(|c| (*c - center).length_squared() <= r2);
}

/// Evaluate queued chunks and spawn their asteroids, bounded per frame by
/// `MAX_CHUNK_EVALS_PER_FRAME` (noise cost) and `MAX_SPAWNS_PER_FRAME`
/// (model build + upload cost). Stale entries that drifted outside the load
/// radius are dropped without being marked visited, so they re-enqueue if
/// the ship returns.
#[allow(clippy::too_many_arguments)]
fn spawn_queued_chunks(
    ship_pos: DVec3,
    chunk_queue: &mut ChunkQueue,
    visited: &mut VisitedChunks,
    loaded: &mut LoadedAsteroids,
    renderer: &mut SceneRenderer,
    sprite_data: &mut SpriteModelRegistry,
    commands: &mut CommandBuffer,
    world_seed: u64,
) {
    if chunk_queue.is_empty() {
        return;
    }

    let center = world_to_chunk(ship_pos);
    let r2 = LOAD_RADIUS * LOAD_RADIUS;
    // All chunks share the world seed; build the permutation table once.
    let perlin = PerlinNoise3D::new(world_seed);

    // Phase 1 — sequential density gate: cheap per chunk, picks this frame's
    // spawn batch so the caps hold exactly (no carry-over between frames).
    let mut to_spawn: Vec<IVec3> = Vec::with_capacity(MAX_SPAWNS_PER_FRAME);
    for _ in 0..MAX_CHUNK_EVALS_PER_FRAME {
        if to_spawn.len() == MAX_SPAWNS_PER_FRAME {
            break;
        }
        let Some(chunk) = chunk_queue.pop_front() else {
            break;
        };
        if (chunk - center).length_squared() > r2 {
            continue;
        }
        visited.0.insert(chunk);
        if chunk_has_asteroid(chunk, world_seed, &perlin) {
            to_spawn.push(chunk);
        }
    }

    // Phase 2 — parallel model builds + kv6 surface conversion (the two
    // expensive steps, ~1.1ms combined per asteroid). par_iter preserves
    // order, so renderer/registry ids stay deterministic.
    let spawn_batch: Vec<(IVec3, SpawnData, Kv6)> = to_spawn
        .into_par_iter()
        .map(|chunk| {
            let data = compute_spawn(chunk, world_seed);
            let kv6 = sprite_model_to_kv6(&data.model);
            (chunk, data, kv6)
        })
        .collect();

    // Phase 3 — sequential GPU upload + ECS spawn.
    for (chunk, data, kv6) in spawn_batch {
        let initial_count = data.model.colors.len() as u32;
        let sprite = spawn_sprite(renderer, sprite_data, data.model, &kv6);
        let entity = commands.push((
            AsteroidMarker,
            AsteroidMinerals {
                points: data.minerals,
            },
            AsteroidVoxelInfo { initial_count },
            Aabb::empty(),
            sprite,
            NewtonBody {
                mass: 1.0,
                pos: data.spawn_pos,
                vel: DVec3::ZERO,
                orientation: DQuat::IDENTITY,
                angular_vel: data.angular_vel,
            },
        ));
        loaded.0.insert(entity, chunk);
    }
}
