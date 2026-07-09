use std::sync::OnceLock;

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
        LoadedAsteroids, PendingCompact, PendingSpawns, VisitedChunks, WorldSeed, CHUNK_SIZE,
        LOAD_RADIUS, UNLOAD_RADIUS,
    },
    systems::sprite::{perform_despawn, spawn_sprite},
};

const UPDATE_DIST_SQ: f64 = (CHUNK_SIZE as f64 / 2.0) * (CHUNK_SIZE as f64 / 2.0);

/// Below this speed (< 1/6 chunk per second) the ship counts as parked:
/// streaming has stopped and won't resume within the next second.
const PARKED_MAX_SPEED: f64 = CHUNK_SIZE as f64 / 6.0;

/// Spawn uploads (kv6 conversion + renderer registration) allowed per frame
/// while flying. Sequential main-thread work — a dense shell holds 50+
/// spawns, and uploading them all in one frame reads as a hitch.
const SPAWN_UPLOADS_MOVING: usize = 8;
/// While parked (startup, after stopping) pop-in speed matters more than
/// frame pacing, so drain the buffer much faster.
const SPAWN_UPLOADS_PARKED: usize = 64;

/// Chunks assigned per rayon thread in each frame's generation batch.
/// Total batch = `CHUNKS_PER_THREAD × num_threads`, capped at `MAX_CHUNK_BATCH`.
const CHUNKS_PER_THREAD: usize = 8;
/// Hard ceiling so high-core machines don't inflate the sequential upload phase.
const MAX_CHUNK_BATCH: usize = 128;

static RAYON_THREAD_COUNT: OnceLock<usize> = OnceLock::new();

/// Call once at startup (before the game loop) to snapshot the rayon pool size.
/// `drain_chunk_queue` self-inits lazily if this is not called, but calling it
/// explicitly front-loads pool creation to a predictable point.
pub fn init_chunk_parallelism() {
    RAYON_THREAD_COUNT.get_or_init(|| rayon::current_num_threads());
}

fn chunk_batch_size() -> usize {
    let threads = RAYON_THREAD_COUNT.get_or_init(|| rayon::current_num_threads());
    (threads * CHUNKS_PER_THREAD).min(MAX_CHUNK_BATCH)
}

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
    #[resource] pending_spawns: &mut PendingSpawns,
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
) {
    let (ship_pos, ship_speed, updated_pos) = {
        let mut query = <(&Miner, &NewtonBody, &mut PresencePosition)>::query();
        let (_, body, presence) = query.iter_mut(world).next().expect("miner missing");
        let pos = body.pos;
        let updated = if pos.distance_squared(presence.0) > UPDATE_DIST_SQ {
            presence.0 = pos;
            true
        } else {
            false
        };
        (pos, body.vel.length(), updated)
    };

    if updated_pos {
        let despawned = update_sprites(
            ship_pos,
            visited,
            loaded,
            renderer,
            sprite_data,
            world,
            commands,
        );
        enqueue_chunks(ship_pos, visited, chunk_queue);
        pending_compact.0 += despawned as u32;
    }

    let upload_cap = if ship_speed < PARKED_MAX_SPEED {
        SPAWN_UPLOADS_PARKED
    } else {
        SPAWN_UPLOADS_MOVING
    };
    drain_chunk_queue(
        ship_pos,
        upload_cap,
        chunk_queue,
        pending_spawns,
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
    // Safety valve: dead entries keep occupancy/color_offsets holes (~2KB per
    // asteroid; colors slots ARE recycled via the free list) that only compact
    // reclaims. 10k dead ≈ 20MB of holes — force a compact past that even
    // while the ship is moving.
    const COMPACT_DEAD_HARD_CAP: u32 = 10_000;
    // Compacting shrinks the GPU buffers to a tight fit, so doing it while
    // asteroids are still streaming guarantees an immediate grow+repack (a
    // second full rebuild). The chunk queue drains within a few frames even
    // in flight, so "queue empty" alone is a bad idle signal — require the
    // ship to be practically parked and the spawn buffer drained as well.
    let should_compact = (pending_compact.0 >= COMPACT_DEAD_THRESHOLD
        && ship_speed < PARKED_MAX_SPEED
        && chunk_queue.is_empty()
        && pending_spawns.0.is_empty())
        || pending_compact.0 >= COMPACT_DEAD_HARD_CAP;

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

/// Compute up to `chunk_batch_size()` chunks per frame (parallel via rayon)
/// into the pending-spawn buffer, then upload at most `upload_cap` spawns
/// (sequential main-thread work) — the throttle that keeps a dense shell
/// from landing as one multi-frame hitch.
/// Prunes entries that have drifted outside the load radius (ship moved away).
#[allow(clippy::too_many_arguments)]
fn drain_chunk_queue(
    ship_pos: DVec3,
    upload_cap: usize,
    chunk_queue: &mut ChunkQueue,
    pending_spawns: &mut PendingSpawns,
    visited: &mut VisitedChunks,
    loaded: &mut LoadedAsteroids,
    renderer: &mut SceneRenderer,
    sprite_data: &mut SpriteModelRegistry,
    commands: &mut CommandBuffer,
    world_seed: u64,
) {
    let center = world_to_chunk(ship_pos);
    let r2 = LOAD_RADIUS * LOAD_RADIUS;

    // Refill the spawn buffer only when it can't cover this frame's uploads,
    // so at most ~one batch of computed models sits waiting.
    if pending_spawns.0.len() < upload_cap && !chunk_queue.is_empty() {
        // Drain stale front entries (ship moved away before they were generated).
        while let Some(&front) = chunk_queue.front() {
            if (front - center).length_squared() > r2 {
                chunk_queue.pop_front();
            } else {
                break;
            }
        }

        if !chunk_queue.is_empty() {
            let batch_size = chunk_batch_size().min(chunk_queue.len());
            let batch = chunk_queue.drain_front(batch_size);

            // Build Perlin noise once for the whole batch — all chunks share the same world seed,
            // so rebuilding the permutation table per chunk is redundant work.
            let perlin = PerlinNoise3D::new(world_seed);

            // Parallel compute phase — par_iter preserves order so chain_id assignment is deterministic.
            let results: Vec<ChunkComputeResult> = batch
                .into_par_iter()
                .map(|chunk| compute_chunk(chunk, world_seed, &perlin))
                .collect();

            for result in results {
                match result {
                    ChunkComputeResult::NoSpawn { chunk } => {
                        visited.0.insert(chunk);
                    }
                    ChunkComputeResult::Spawn(data) => {
                        // Mark visited now so a presence update can't re-enqueue
                        // the chunk while its spawn waits in the buffer; the
                        // stale-drop below un-visits it.
                        visited.0.insert(data.chunk);
                        pending_spawns.0.push_back(data);
                    }
                }
            }
        }
    }

    // Throttled upload phase: GPU registry access and ECS command buffer are
    // single-threaded, so this is the per-frame cost we're bounding.
    for _ in 0..upload_cap {
        let Some(data) = pending_spawns.0.pop_front() else {
            break;
        };
        if (data.chunk - center).length_squared() > r2 {
            // Ship moved away while this spawn waited — drop it and make the
            // chunk re-populatable.
            visited.0.remove(&data.chunk);
            continue;
        }
        let initial_count = data.model.colors.len() as u32;
        let sprite = spawn_sprite(renderer, sprite_data, data.model);
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
        loaded.0.insert(entity);
    }
}

/// Single pass over all loaded asteroids: fully despawn those that left the presence radius.
/// Returns the number of asteroids despawned.
fn update_sprites(
    ship_pos: DVec3,
    visited: &mut VisitedChunks,
    loaded: &mut LoadedAsteroids,
    renderer: &mut SceneRenderer,
    registry: &mut SpriteModelRegistry,
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
) -> usize {
    let center = world_to_chunk(ship_pos);
    let r2 = UNLOAD_RADIUS * UNLOAD_RADIUS;

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
        perform_despawn(entity, world, commands, renderer, registry);
        loaded.0.remove(&entity);
        visited.0.remove(&chunk);
    }
    despawn_count
}
