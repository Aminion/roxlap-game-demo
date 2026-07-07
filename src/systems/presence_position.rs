use glam::{DQuat, DVec3, IVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, *};
use rayon::prelude::*;
use roxlap_cavegen::PerlinNoise3D;
use roxlap_gpu::SpriteModelRegistry;
use roxlap_render::SceneRenderer;

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
        compute_chunk, missing_chunks, world_to_chunk, ChunkComputeResult, ChunkQueue,
        LoadedAsteroids, PendingCompact, VisitedChunks, WorldSeed, CHUNK_SIZE, LOAD_RADIUS,
    },
    systems::sprite::{perform_despawn, spawn_sprite},
};

const UPDATE_DIST_SQ: f64 = (CHUNK_SIZE as f64 / 2.0) * (CHUNK_SIZE as f64 / 2.0);

/// Number of chunks pulled from the queue and processed per frame.
/// The compute phase runs in parallel, so wall time ≈ single-chunk cost / thread count.
const CHUNK_BATCH_SIZE: usize = 64;

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[write_component(PresencePosition)]
#[write_component(Sprite)]
pub fn presence_position_update(
    #[resource] visited: &mut VisitedChunks,
    #[resource] loaded: &mut LoadedAsteroids,
    #[resource] renderer: &mut SceneRenderer,
    #[resource] sprite_data: &mut SpriteModelRegistry,
    #[resource] world_seed: &WorldSeed,
    #[resource] pending_compact: &mut PendingCompact,
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
        let despawned = update_sprites(ship_pos, visited, loaded, renderer, world, commands);
        enqueue_chunks(ship_pos, visited, chunk_queue);
        pending_compact.0 += despawned as u32;
    }

    drain_chunk_queue(
        ship_pos,
        chunk_queue,
        visited,
        loaded,
        renderer,
        sprite_data,
        commands,
        world_seed.0,
    );
    // Compact when dead models accumulate past this threshold. Cost is O(live volume)
    // not O(dead count), so firing every unload cycle (94–165 dead, ~300KB recovered)
    // is wasteful — defer until the waste is worth a 35–66ms rebuild.
    const COMPACT_DEAD_THRESHOLD: u32 = 300;

    let should_compact = pending_compact.0 >= COMPACT_DEAD_THRESHOLD;

    if should_compact {
        renderer.compact_sprite_models();
        pending_compact.0 = 0;
    }
}

/// Enqueue all chunks within load radius that are neither visited nor already queued.
fn enqueue_chunks(ship_pos: DVec3, visited: &VisitedChunks, chunk_queue: &mut ChunkQueue) {
    for chunk in missing_chunks(ship_pos, LOAD_RADIUS, &visited.0) {
        // enqueue() is a no-op if the chunk is already in the queue set.
        chunk_queue.enqueue(chunk);
    }
}

/// Drain up to `CHUNK_BATCH_SIZE` chunks per frame.
/// Compute phase runs in parallel via rayon; GPU upload is sequential on the main thread.
/// Prunes entries that have drifted outside the load radius (ship moved away).
fn drain_chunk_queue(
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

    // Drain stale front entries (ship moved away before they were generated).
    while let Some(&front) = chunk_queue.front() {
        if (front - center).length_squared() > r2 {
            chunk_queue.pop_front();
        } else {
            break;
        }
    }

    if chunk_queue.is_empty() {
        return;
    }

    let batch_size = CHUNK_BATCH_SIZE.min(chunk_queue.len());
    let batch = chunk_queue.drain_front(batch_size);

    // Build Perlin noise once for the whole batch — all chunks share the same world seed,
    // so rebuilding the permutation table per chunk is redundant work.
    let perlin = PerlinNoise3D::new(world_seed);

    // Parallel compute phase — par_iter preserves order so chain_id assignment is deterministic.
    let results: Vec<ChunkComputeResult> = batch
        .into_par_iter()
        .map(|chunk| compute_chunk(chunk, world_seed, &perlin))
        .collect();

    // Sequential upload phase: GPU registry access and ECS command buffer are single-threaded.
    for result in results {
        match result {
            ChunkComputeResult::NoSpawn { chunk } => {
                visited.0.insert(chunk);
            }
            ChunkComputeResult::Spawn {
                chunk,
                model,
                minerals,
                spawn_pos,
                angular_vel,
            } => {
                visited.0.insert(chunk);
                let initial_count = model.colors.len() as u32;
                let sprite = spawn_sprite(renderer, sprite_data, model);
                let entity = commands.push((
                    AsteroidMarker,
                    AsteroidMinerals { points: minerals },
                    AsteroidVoxelInfo { initial_count },
                    Aabb::empty(),
                    sprite,
                    NewtonBody {
                        mass: 1.0,
                        pos: spawn_pos,
                        vel: DVec3::ZERO,
                        orientation: DQuat::IDENTITY,
                        angular_vel,
                    },
                ));
                loaded.0.insert(entity);
            }
        }
    }
}

/// Single pass over all loaded asteroids: fully despawn those that left the presence radius.
/// Returns the number of asteroids despawned.
fn update_sprites(
    ship_pos: DVec3,
    visited: &mut VisitedChunks,
    loaded: &mut LoadedAsteroids,
    renderer: &mut SceneRenderer,
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
) -> usize {
    let center = world_to_chunk(ship_pos);
    let r2 = LOAD_RADIUS * LOAD_RADIUS;

    let mut to_unload: Vec<(Entity, IVec3)> = Vec::new();
    for &entity in &loaded.0 {
        let Ok(entry) = world.entry_ref(entity) else {
            continue;
        };
        let Ok(body) = entry.get_component::<NewtonBody>() else {
            continue;
        };
        let chunk = world_to_chunk(body.pos);
        if (chunk - center).length_squared() > r2 {
            to_unload.push((entity, chunk));
        }
    }

    if to_unload.is_empty() {
        return 0;
    }

    let despawn_count = to_unload.len();
    for (entity, chunk) in to_unload {
        perform_despawn(entity, world, commands, renderer);
        loaded.0.remove(&entity);
        visited.0.remove(&chunk);
    }
    despawn_count
}
