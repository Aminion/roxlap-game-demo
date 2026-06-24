use std::collections::HashMap;

use glam::{DQuat, DVec3, IVec3, UVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, EntityStore, *};
use rayon::prelude::*;
use roxlap_cavegen::PerlinNoise3D;
use roxlap_gpu::{GpuRenderer, SpriteModel};

use crate::{
    components::{
        aabb::Aabb,
        asteroid::{AsteroidMarker, AsteroidMinerals, AsteroidVoxelInfo},
        miner::Miner,
        newton_body::NewtonBody,
        presence_position::PresencePosition,
        sprite_id::Sprite,
    },
    generation::chunks::{missing_chunks, world_to_chunk, CHUNK_SIZE, LOAD_RADIUS},
    math::{hash_to_signed, splitmix64},
    world::{build_asteroid, spawn_sprite, ASTEROID_VOXEL_SIZE},
    ChunkQueue, LoadedAsteroids, PendingCompact, QueuedChunks, SpriteData, VisitedChunks,
    WorldSeed,
};

const UPDATE_DIST_SQ: f64 = (CHUNK_SIZE as f64 / 2.0) * (CHUNK_SIZE as f64 / 2.0);

/// Base spatial frequency of the density noise. 1/0.03 ≈ 33 chunks per noise
/// wavelength — large-scale void/dense structure roughly twice the load sphere.
const CHUNK_NOISE_FREQ: f32 = 0.03;

/// fBm octave count. Each octave doubles the frequency, adding finer patchiness:
/// octave 1 ≈ 33-chunk blobs, octave 2 ≈ 16-chunk, octave 3 ≈ 8-chunk.
const CHUNK_NOISE_OCTAVES: u32 = 3;

/// Perlin outputs ≈ ±0.866 (theoretical max √3/2); divide by this to normalise to ±1.
const PERLIN_MAX: f32 = 0.866;

/// Number of chunks pulled from the queue and processed per frame.
/// The compute phase runs in parallel, so wall time ≈ single-chunk cost / thread count.
const CHUNK_BATCH_SIZE: usize = 32;

/// Fraction of asteroids that contain crystal deposits. Range [0.0, 1.0].
const CRYSTAL_SPAWN_CHANCE: f32 = 0.01;

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
    let queue_now_empty = chunk_queue.0.is_empty();

    // Compact when dead models accumulate past this threshold. Cost is O(live volume)
    // not O(dead count), so firing every unload cycle (94–165 dead, ~300KB recovered)
    // is wasteful — defer until the waste is worth a 35–66ms rebuild.
    const COMPACT_DEAD_THRESHOLD: u32 = 300;

    let should_compact = queue_now_empty && pending_compact.0 >= COMPACT_DEAD_THRESHOLD;

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
    let to_enqueue: Vec<_> = missing_chunks(ship_pos, LOAD_RADIUS, &visited.0)
        .filter(|c| !queued_chunks.0.contains(c))
        .collect();
    for chunk in to_enqueue {
        // Mark as queued immediately so subsequent threshold crossings don't re-enqueue.
        // `visited` is only updated when the chunk is actually generated.
        chunk_queue.0.push_back(chunk);
        queued_chunks.0.insert(chunk);
    }
}

enum ChunkComputeResult {
    NoSpawn {
        chunk: IVec3,
    },
    Spawn {
        chunk: IVec3,
        model: SpriteModel,
        minerals: Vec<UVec3>,
        spawn_pos: DVec3,
        angular_vel: DVec3,
    },
}

/// Pure-CPU chunk evaluation: density noise + optional asteroid model building.
/// No GPU or ECS access — safe to call from rayon threads.
fn compute_chunk(chunk: IVec3, world_seed: u64) -> ChunkComputeResult {
    let perlin = PerlinNoise3D::new(world_seed);
    let raw = perlin.fbm(
        chunk.x as f32,
        chunk.y as f32,
        chunk.z as f32,
        CHUNK_NOISE_OCTAVES,
        CHUNK_NOISE_FREQ,
    );
    let density = ((raw / PERLIN_MAX) + 1.0) * 0.5;
    let density = density.clamp(0.0, 1.0);
    let density = density * density * (3.0 - 2.0 * density);

    if chunk_spawn_hash(world_seed, chunk) >= density {
        return ChunkComputeResult::NoSpawn { chunk };
    }

    let h = chunk_hash_base(world_seed, chunk);
    let chunk_centre = (chunk.as_dvec3() + DVec3::splat(0.5)) * CHUNK_SIZE as f64;
    let spawn_pos = chunk_centre + chunk_spawn_offset(h);
    let has_crystals = (splitmix64(h.wrapping_add(8)) >> 40) as f32 / 16_777_216.0 < CRYSTAL_SPAWN_CHANCE;
    let noise_seed = h.wrapping_add(9);
    let scale_seed = h.wrapping_add(10);
    let (model, minerals) = build_asteroid(
        h.wrapping_add(6),
        h.wrapping_add(7),
        noise_seed,
        scale_seed,
        has_crystals,
    );
    let angular_vel = chunk_spawn_angular_vel(h);

    ChunkComputeResult::Spawn {
        chunk,
        model,
        minerals,
        spawn_pos,
        angular_vel,
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
                    Aabb {
                        half_extent: ASTEROID_VOXEL_SIZE as f32 / 2.0,
                    },
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

/// Combine world seed and chunk coordinates into one u64 base hash.
/// Each axis is multiplied by a distinct irrational-derived constant
/// (φ, π, e scaled to u64) so that chunks differing on only one axis
/// produce unrelated sums before the splitmix64 avalanche pass.
fn chunk_hash_base(world_seed: u64, chunk: IVec3) -> u64 {
    splitmix64(
        world_seed
            .wrapping_add((chunk.x as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .wrapping_add((chunk.y as u64).wrapping_mul(0x6c62272e07bb0142))
            .wrapping_add((chunk.z as u64).wrapping_mul(0x4d2c6dfc5ac42aad)),
    )
}

fn chunk_spawn_hash(world_seed: u64, chunk: IVec3) -> f32 {
    let h = chunk_hash_base(world_seed, chunk);
    // Top 24 bits → [0, 1)
    (h >> 40) as f32 / 16_777_216.0
}

/// Max axis offset from chunk centre: CHUNK_SIZE/2 − half_extent = 32 − 8 = 24.
/// Worst-case gap between adjacent asteroids = 64 − 2×24 = 16 = 2×half_extent (touching, not overlapping).
const SPAWN_SAFE_RANGE: f64 = 24.0;

fn chunk_spawn_offset(h: u64) -> DVec3 {
    // Axis indices 0/1/2: each splitmix64 call decorrelates adjacent seed values.
    DVec3::from([0u64, 1, 2].map(|i| hash_to_signed(splitmix64(h.wrapping_add(i)))))
        * SPAWN_SAFE_RANGE
}

fn chunk_spawn_angular_vel(h: u64) -> DVec3 {
    // Indices 3/4/5 are independent from the position offset indices 0/1/2.
    DVec3::from([3u64, 4, 5].map(|i| hash_to_signed(splitmix64(h.wrapping_add(i)))))
}

/// Bidirectional slot↔entity index kept in sync with GPU sprite storage.
pub struct SpriteMaps {
    pub slot_to_entity: HashMap<u32, Entity>,
    pub entity_to_slot: HashMap<Entity, u32>,
}

/// Update `maps` after a GPU swap-remove moved `displaced_old` → `current_slot`.
/// Returns the entity whose `Sprite` needs updating to `current_slot`, if any.
fn apply_swap_remove(
    current_slot: u32,
    displaced_old: u32,
    maps: &mut SpriteMaps,
) -> Option<Entity> {
    let displaced_entity = maps.slot_to_entity.remove(&displaced_old)?;
    maps.entity_to_slot.insert(displaced_entity, current_slot);
    maps.slot_to_entity.insert(current_slot, displaced_entity);
    Some(displaced_entity)
}

/// Remove a single entity from the GPU, ECS, and all bookkeeping structures.
///
/// `maps` must cover every currently-loaded sprite entity before the first call
/// in a batch; it is updated in-place so sequential calls within the same batch
/// stay consistent.
///
/// Does NOT touch `VisitedChunks` — callers that want the chunk to be re-populatable
/// (distance-based unload) must remove it from `visited` themselves.
pub fn perform_despawn(
    entity: Entity,
    maps: &mut SpriteMaps,
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    gpu: &mut GpuRenderer,
) {
    let chain_id = match world.entry_ref(entity) {
        Ok(e) => match e.get_component::<Sprite>() {
            Ok(s) => s.chain_id,
            Err(_) => return,
        },
        Err(_) => return,
    };

    let current_slot = match maps.entity_to_slot.remove(&entity) {
        Some(s) => s,
        None => return,
    };
    maps.slot_to_entity.remove(&current_slot);

    if let Some(displaced_old) = gpu.remove_sprite_instance(current_slot as usize) {
        if let Some(displaced_entity) = apply_swap_remove(current_slot, displaced_old as u32, maps)
        {
            if let Ok(mut entry) = world.entry_mut(displaced_entity) {
                if let Ok(sprite) = entry.get_component_mut::<Sprite>() {
                    sprite.slot = current_slot;
                }
            }
        }
    }

    gpu.remove_sprite_model(chain_id);
    commands.remove(entity);
}

/// Build a `SpriteMaps` covering every entity with a `Sprite`.
pub fn build_sprite_maps(world: &mut SubWorld) -> SpriteMaps {
    let mut slot_to_entity: HashMap<u32, Entity> = HashMap::new();
    let mut entity_to_slot: HashMap<Entity, u32> = HashMap::new();
    let mut q = <(Entity, &Sprite)>::query();
    for (&entity, sprite) in q.iter(world) {
        slot_to_entity.insert(sprite.slot, entity);
        entity_to_slot.insert(entity, sprite.slot);
    }
    SpriteMaps {
        slot_to_entity,
        entity_to_slot,
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
    // Covers all sprite entities so swap-removes triggered by asteroid despawns
    // correctly update any displaced entity (including projectiles/crystals).
    let mut maps = build_sprite_maps(world);

    // Decide which asteroids are out of range.
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

    let despawn_count = to_unload.len();
    for (entity, chunk) in to_unload {
        perform_despawn(entity, &mut maps, world, commands, gpu);
        loaded.0.remove(&entity);
        visited.0.remove(&chunk);
    }
    despawn_count
}

#[cfg(test)]
mod tests {
    use super::{
        apply_swap_remove, chunk_hash_base, chunk_spawn_angular_vel, chunk_spawn_hash,
        chunk_spawn_offset, SpriteMaps, SPAWN_SAFE_RANGE,
    };
    use crate::generation::chunks::CHUNK_SIZE;
    use glam::{DVec3, IVec3};
    use legion::Entity;
    use std::collections::HashMap;

    fn make_entities(n: usize) -> (legion::World, Vec<Entity>) {
        let mut world = legion::World::default();
        let entities: Vec<Entity> = (0..n).map(|_| world.push((0u8,))).collect();
        (world, entities)
    }

    // ── chunk_spawn_hash ──────────────────────────────────────────────────────

    #[test]
    fn chunk_hash_in_unit_range() {
        for seed in [0u64, 1, u64::MAX, 0xdead_beef] {
            for chunk in [
                IVec3::ZERO,
                IVec3::new(1, -1, 1000),
                IVec3::new(-100, 200, -300),
            ] {
                let v = chunk_spawn_hash(seed, chunk);
                assert!(
                    (0.0..1.0).contains(&v),
                    "hash out of [0,1): {v} for chunk {chunk} seed {seed}"
                );
            }
        }
    }

    #[test]
    fn chunk_hash_differs_by_seed() {
        let chunk = IVec3::new(5, 5, 5);
        assert_ne!(
            chunk_spawn_hash(0, chunk),
            chunk_spawn_hash(1, chunk),
            "different seeds must produce different hashes"
        );
    }

    #[test]
    fn chunk_hash_differs_by_coord() {
        let seed = 42u64;
        let hx = chunk_spawn_hash(seed, IVec3::new(1, 0, 0));
        let hy = chunk_spawn_hash(seed, IVec3::new(0, 1, 0));
        let hz = chunk_spawn_hash(seed, IVec3::new(0, 0, 1));
        let hn = chunk_spawn_hash(seed, IVec3::new(-1, 0, 0));
        assert_ne!(hx, hy);
        assert_ne!(hy, hz);
        assert_ne!(hx, hn, "positive and negative coords must hash differently");
    }

    // ── chunk_spawn_angular_vel ───────────────────────────────────────────────

    #[test]
    fn spawn_angular_vel_in_unit_range() {
        for seed in [0u64, 1, 0xdead_beef, u64::MAX] {
            for chunk in [
                IVec3::ZERO,
                IVec3::new(1, -1, 1000),
                IVec3::new(-100, 200, -300),
            ] {
                let v = chunk_spawn_angular_vel(chunk_hash_base(seed, chunk));
                assert!(
                    v.x.abs() <= 1.0 && v.y.abs() <= 1.0 && v.z.abs() <= 1.0,
                    "angular_vel {v} out of [-1,1] for chunk {chunk} seed {seed}"
                );
            }
        }
    }

    // ── chunk_spawn_offset ────────────────────────────────────────────────────

    #[test]
    fn spawn_offset_within_safe_range() {
        for seed in [0u64, 1, 0xdead_beef, u64::MAX] {
            for chunk in [
                IVec3::ZERO,
                IVec3::new(1, -1, 1000),
                IVec3::new(-100, 200, -300),
            ] {
                let o = chunk_spawn_offset(chunk_hash_base(seed, chunk));
                assert!(
                    o.x.abs() <= SPAWN_SAFE_RANGE
                        && o.y.abs() <= SPAWN_SAFE_RANGE
                        && o.z.abs() <= SPAWN_SAFE_RANGE,
                    "offset {o} exceeds safe range for chunk {chunk} seed {seed}"
                );
            }
        }
    }

    #[test]
    fn adjacent_asteroids_do_not_overlap() {
        // Worst case: both asteroids offset maximally toward the shared boundary.
        // Asteroid A in chunk (0,0,0) offset +SAFE in X; B in chunk (1,0,0) offset -SAFE in X.
        // Gap must be >= 2 * half_extent (16) for no AABB overlap.
        let half_extent = 8.0_f64;
        let chunk_a = IVec3::new(0, 0, 0);
        let chunk_b = IVec3::new(1, 0, 0);
        let centre_a = (chunk_a.as_dvec3() + DVec3::splat(0.5)) * CHUNK_SIZE as f64;
        let centre_b = (chunk_b.as_dvec3() + DVec3::splat(0.5)) * CHUNK_SIZE as f64;
        let worst_a = centre_a + DVec3::new(SPAWN_SAFE_RANGE, 0.0, 0.0);
        let worst_b = centre_b - DVec3::new(SPAWN_SAFE_RANGE, 0.0, 0.0);
        let gap = (worst_b.x - worst_a.x).abs();
        assert!(
            gap >= 2.0 * half_extent,
            "gap {gap} < min separation {}",
            2.0 * half_extent
        );
    }

    // ── apply_swap_remove ─────────────────────────────────────────────────────

    #[test]
    fn swap_remove_missing_displaced_returns_none() {
        let (_world, entities) = make_entities(1);
        let e0 = entities[0];
        let mut maps = SpriteMaps {
            slot_to_entity: HashMap::from([(0, e0)]),
            entity_to_slot: HashMap::from([(e0, 0)]),
        };
        // displaced_old=99 is absent — nothing to reassign.
        assert!(apply_swap_remove(0, 99, &mut maps).is_none());
    }

    #[test]
    fn swap_remove_normal_swap() {
        // Entities at slots 0, 1, 2. Remove slot 0; GPU moves e2 (slot 2) → slot 0.
        let (_world, entities) = make_entities(3);
        let [e0, e1, e2] = [entities[0], entities[1], entities[2]];
        let mut maps = SpriteMaps {
            slot_to_entity: HashMap::from([(0u32, e0), (1, e1), (2, e2)]),
            entity_to_slot: HashMap::from([(e0, 0u32), (e1, 1), (e2, 2)]),
        };

        // Caller removes the departing entity before calling apply_swap_remove.
        maps.slot_to_entity.remove(&0);
        maps.entity_to_slot.remove(&e0);

        let result = apply_swap_remove(0, 2, &mut maps);

        assert_eq!(result, Some(e2));
        assert_eq!(maps.slot_to_entity[&0], e2, "slot 0 now maps to e2");
        assert_eq!(maps.entity_to_slot[&e2], 0, "e2 now tracks slot 0");
        assert!(
            !maps.slot_to_entity.contains_key(&2),
            "old slot 2 must be vacated"
        );
        assert_eq!(maps.slot_to_entity[&1], e1, "e1 at slot 1 is untouched");
    }

    #[test]
    fn swap_remove_sequential_unloads() {
        // 4 entities. Unload slot 0 (e3 swaps 3→0), then unload slot 1 (e2 swaps 2→1).
        let (_world, entities) = make_entities(4);
        let [e0, e1, e2, e3] = [entities[0], entities[1], entities[2], entities[3]];
        let mut maps = SpriteMaps {
            slot_to_entity: HashMap::from([(0u32, e0), (1, e1), (2, e2), (3, e3)]),
            entity_to_slot: HashMap::from([(e0, 0u32), (e1, 1), (e2, 2), (e3, 3)]),
        };

        // First unload: e0 at slot 0; GPU moves e3 (slot 3) → slot 0.
        maps.slot_to_entity.remove(&0);
        maps.entity_to_slot.remove(&e0);
        assert_eq!(apply_swap_remove(0, 3, &mut maps), Some(e3));

        // Second unload: e1 at slot 1; GPU moves e2 (slot 2) → slot 1.
        maps.slot_to_entity.remove(&1);
        maps.entity_to_slot.remove(&e1);
        assert_eq!(apply_swap_remove(1, 2, &mut maps), Some(e2));

        assert_eq!(maps.entity_to_slot[&e3], 0);
        assert_eq!(maps.entity_to_slot[&e2], 1);
        assert_eq!(maps.slot_to_entity[&0], e3);
        assert_eq!(maps.slot_to_entity[&1], e2);
        assert!(
            !maps.slot_to_entity.contains_key(&2),
            "slot 2 must be vacated"
        );
        assert!(
            !maps.slot_to_entity.contains_key(&3),
            "slot 3 must be vacated"
        );
    }
}
