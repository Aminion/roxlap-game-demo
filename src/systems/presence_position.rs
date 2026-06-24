use glam::{DQuat, DVec3, IVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, *};
use rayon::prelude::*;
use roxlap_gpu::GpuRenderer;

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
        compute_chunk, missing_chunks, world_to_chunk, ChunkComputeResult, CHUNK_SIZE, LOAD_RADIUS,
    },
    systems::sprite::{build_sprite_maps, perform_despawn},
    world::spawn_sprite,
    ChunkQueue, LoadedAsteroids, PendingCompact, QueuedChunks, SpriteData, VisitedChunks,
    WorldSeed,
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
    #[resource] gpu: &mut GpuRenderer,
    #[resource] sprite_data: &mut SpriteData,
    #[resource] world_seed: &WorldSeed,
    #[resource] pending_compact: &mut PendingCompact,
    #[resource] chunk_queue: &mut ChunkQueue,
    #[resource] queued_chunks: &mut QueuedChunks,
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
        let despawned = update_sprites(ship_pos, visited, loaded, gpu, world, commands);
        enqueue_chunks(ship_pos, visited, queued_chunks, chunk_queue);
        pending_compact.0 += despawned as u32;
    }

    drain_chunk_queue(
        ship_pos,
        chunk_queue,
        queued_chunks,
        visited,
        loaded,
        gpu,
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
        gpu.compact_sprite_models(&sprite_data.registry);
        pending_compact.0 = 0;
    }
}

/// Enqueue all chunks within load radius that are neither visited nor already queued.
fn enqueue_chunks(
    ship_pos: DVec3,
    visited: &VisitedChunks,
    queued_chunks: &mut QueuedChunks,
    chunk_queue: &mut ChunkQueue,
) {
    for chunk in missing_chunks(ship_pos, LOAD_RADIUS, &visited.0) {
        if !queued_chunks.0.contains(&chunk) {
            // Mark as queued immediately so subsequent threshold crossings don't re-enqueue.
            // `visited` is only updated when the chunk is actually generated.
            chunk_queue.0.push_back(chunk);
            queued_chunks.0.insert(chunk);
        }
    }
}

/// Drain up to `CHUNK_BATCH_SIZE` chunks per frame.
/// Compute phase runs in parallel via rayon; GPU upload is sequential on the main thread.
/// Prunes entries that have drifted outside the load radius (ship moved away).
fn drain_chunk_queue(
    ship_pos: DVec3,
    chunk_queue: &mut ChunkQueue,
    queued_chunks: &mut QueuedChunks,
    visited: &mut VisitedChunks,
    loaded: &mut LoadedAsteroids,
    gpu: &mut GpuRenderer,
    sprite_data: &mut SpriteData,
    commands: &mut CommandBuffer,
    world_seed: u64,
) {
    if chunk_queue.0.is_empty() {
        return;
    }

    let center = world_to_chunk(ship_pos);
    let r2 = LOAD_RADIUS * LOAD_RADIUS;

    // Drain stale front entries (ship moved away before they were generated).
    while let Some(&front) = chunk_queue.0.front() {
        if (front - center).length_squared() > r2 {
            chunk_queue.0.pop_front();
            queued_chunks.0.remove(&front);
        } else {
            break;
        }
    }

    if chunk_queue.0.is_empty() {
        return;
    }

    let batch_size = CHUNK_BATCH_SIZE.min(chunk_queue.0.len());
    let batch: Vec<IVec3> = chunk_queue.0.drain(..batch_size).collect();
    for &chunk in &batch {
        queued_chunks.0.remove(&chunk);
    }

    // Parallel compute phase — par_iter preserves order so chain_id assignment is deterministic.
    let results: Vec<ChunkComputeResult> = batch
        .into_par_iter()
        .map(|chunk| compute_chunk(chunk, world_seed))
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
                let sprite = spawn_sprite(&mut sprite_data.registry, gpu, model);
                let initial_count = sprite_data.registry.model(sprite.chain_id).colors.len() as u32;
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
    gpu: &mut GpuRenderer,
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

    // Covers all sprite entities so swap-removes triggered by asteroid despawns
    // correctly update any displaced entity (including projectiles/crystals).
    let mut maps = build_sprite_maps(world);
    let despawn_count = to_unload.len();
    for (entity, chunk) in to_unload {
        perform_despawn(entity, &mut maps, world, commands, gpu);
        loaded.0.remove(&entity);
        visited.0.remove(&chunk);
    }
    despawn_count
}
