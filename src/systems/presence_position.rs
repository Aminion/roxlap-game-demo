use std::collections::HashMap;

use glam::{DQuat, DVec3, IVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, EntityStore, *};
use roxlap_cavegen::PerlinNoise3D;
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
    generation::chunks::{missing_chunks, world_to_chunk, CHUNK_SIZE, LOAD_RADIUS},
    world::{
        build_asteroid_sprite_model, generate_mineral_points, spawn_sprite, ASTEROID_VOXEL_SIZE,
    },
    LoadedAsteroids, SpriteData, VisitedChunks, WorldSeed,
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
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
) {
    let updated_pos = {
        let mut query = <(&Miner, &NewtonBody, &mut PresencePosition)>::query();
        let (_, body, presence) = query.iter_mut(world).next().expect("miner missing");
        if body.pos.distance_squared(presence.0) > UPDATE_DIST_SQ {
            presence.0 = body.pos;
            Some(body.pos)
        } else {
            None
        }
    };

    if let Some(ship_pos) = updated_pos {
        update_sprites(ship_pos, visited, loaded, gpu, world, commands);
        populate_chunks(
            ship_pos,
            visited,
            loaded,
            gpu,
            sprite_data,
            commands,
            world_seed.0,
        );
        // Reclaim GPU VRAM from tombstoned asteroid models. Chain indices are
        // preserved by compact so existing Sprite.chain_id values remain valid.
        gpu.compact_sprite_models(&sprite_data.registry);
    }
}

fn splitmix64(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    h
}

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

/// Maps the top 24 bits of a hash word to [-1, 1).
fn hash_to_signed(v: u64) -> f64 {
    (v >> 40) as f64 / 8_388_608.0 - 1.0
}

fn chunk_spawn_offset(world_seed: u64, chunk: IVec3) -> DVec3 {
    let h = chunk_hash_base(world_seed, chunk);
    // Axis indices 0/1/2: each splitmix64 call decorrelates adjacent seed values.
    DVec3::from([0u64, 1, 2].map(|i| hash_to_signed(splitmix64(h.wrapping_add(i)))))
        * SPAWN_SAFE_RANGE
}

fn chunk_spawn_angular_vel(world_seed: u64, chunk: IVec3) -> DVec3 {
    let h = chunk_hash_base(world_seed, chunk);
    // Indices 3/4/5 are independent from the position offset indices 0/1/2.
    DVec3::from([3u64, 4, 5].map(|i| hash_to_signed(splitmix64(h.wrapping_add(i)))))
}

fn populate_chunks(
    ship_pos: DVec3,
    visited: &mut VisitedChunks,
    loaded: &mut LoadedAsteroids,
    gpu: &mut GpuRenderer,
    sprite_data: &mut SpriteData,
    commands: &mut CommandBuffer,
    world_seed: u64,
) {
    let to_generate: Vec<_> = missing_chunks(ship_pos, LOAD_RADIUS, &visited.0).collect();

    if to_generate.is_empty() {
        return;
    }

    let perlin = PerlinNoise3D::new(world_seed);

    for chunk in to_generate {
        // Sample regional density: normalise Perlin's ±0.866 output to [0, 1].
        let raw = perlin.fbm(
            chunk.x as f32,
            chunk.y as f32,
            chunk.z as f32,
            CHUNK_NOISE_OCTAVES,
            CHUNK_NOISE_FREQ,
        );
        let density = ((raw / PERLIN_MAX) + 1.0) * 0.5;
        let density = density.clamp(0.0, 1.0);
        // Smoothstep S-curve: steepens void/dense boundaries without shifting the midpoint.
        let density = density * density * (3.0 - 2.0 * density);

        visited.0.insert(chunk);

        // Deterministic per-chunk spawn decision: skip if hash falls outside density.
        if chunk_spawn_hash(world_seed, chunk) >= density {
            continue;
        }

        let chunk_centre = (chunk.as_dvec3() + DVec3::splat(0.5)) * CHUNK_SIZE as f64;
        let spawn_pos = chunk_centre + chunk_spawn_offset(world_seed, chunk);
        let h = chunk_hash_base(world_seed, chunk);
        // Top bit of hash index 8 gates crystal presence (~50 % of asteroids).
        let has_crystals = splitmix64(h.wrapping_add(8)) >> 63 == 0;
        let noise_seed = h.wrapping_add(9);
        let scale_seed = h.wrapping_add(10);
        let minerals = if has_crystals {
            generate_mineral_points(
                ASTEROID_VOXEL_SIZE,
                h.wrapping_add(7),
                noise_seed,
                scale_seed,
            )
        } else {
            vec![]
        };
        let sprite = spawn_sprite(
            &mut sprite_data.registry,
            gpu,
            build_asteroid_sprite_model(h.wrapping_add(6), noise_seed, scale_seed, minerals.len()),
        );
        let initial_count = sprite_data.registry.model(sprite.chain_id).colors.len() as u32;
        let angular_vel = chunk_spawn_angular_vel(world_seed, chunk);
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
fn update_sprites(
    ship_pos: DVec3,
    visited: &mut VisitedChunks,
    loaded: &mut LoadedAsteroids,
    gpu: &mut GpuRenderer,
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
) {
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
        let d = chunk - center;
        if d.dot(d) > r2 {
            to_unload.push((entity, chunk));
        }
    }

    for (entity, chunk) in to_unload {
        perform_despawn(entity, &mut maps, world, commands, gpu);
        loaded.0.remove(&entity);
        visited.0.remove(&chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_swap_remove, chunk_spawn_angular_vel, chunk_spawn_hash, chunk_spawn_offset,
        SpriteMaps, SPAWN_SAFE_RANGE,
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
                let v = chunk_spawn_angular_vel(seed, chunk);
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
                let o = chunk_spawn_offset(seed, chunk);
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
