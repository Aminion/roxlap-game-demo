use std::collections::HashMap;

use bytemuck::Zeroable;
use glam::{DQuat, DVec3, IVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, EntityStore, *};
use rand::RngExt;
use roxlap_cavegen::PerlinNoise3D;
use roxlap_gpu::{GpuRenderer, SpriteInstance, SpriteInstanceTransform};

use crate::{
    components::{
        asteroid::{AsteroidChainId, AsteroidMarker},
        miner::Miner,
        newton_body::NewtonBody,
        presence_position::PresencePosition,
        sprite_id::SpriteId,
    },
    generation::chunks::{missing_chunks, world_to_chunk, CHUNK_SIZE, LOAD_RADIUS},
    world::build_asteroid_sprite_model,
    LoadedAsteroids, SpriteData, VisitedChunks, WorldSeed,
};

const UPDATE_DIST_SQ: f64 = (CHUNK_SIZE as f64 / 2.0) * (CHUNK_SIZE as f64 / 2.0);

/// Spatial frequency of the density noise — lower = larger void/dense blobs.
/// freq = 0.5 / desired_blob_diameter_in_chunks. At 0.03, blobs are ~16 chunks
/// across, matching the load-sphere diameter (2 × LOAD_RADIUS = 16).
const CHUNK_NOISE_FREQ: f32 = 0.03;

/// Perlin outputs ≈ ±0.866 (theoretical max √3/2); divide by this to normalise to ±1.
const PERLIN_MAX: f32 = 0.866;

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[read_component(AsteroidChainId)]
#[write_component(PresencePosition)]
#[write_component(SpriteId)]
pub fn presence_position_update(
    #[resource] visited: &mut VisitedChunks,
    #[resource] loaded: &mut LoadedAsteroids,
    #[resource] gpu: &mut GpuRenderer,
    #[resource] sprite_data: &mut SpriteData,
    #[resource] world_seed: &WorldSeed,
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
    }
}

fn chunk_spawn_hash(world_seed: u64, chunk: IVec3) -> f32 {
    // Mix world seed with per-axis coords so negative and positive chunks hash distinctly.
    let mut h = world_seed
        .wrapping_add((chunk.x as i64 as u64).wrapping_mul(0x9e3779b97f4a7c15))
        .wrapping_add((chunk.y as i64 as u64).wrapping_mul(0x6c62272e07bb0142))
        .wrapping_add((chunk.z as i64 as u64).wrapping_mul(0x4d2c6dfc5ac42aad));
    // SplitMix64 finalizer — avalanches all bits
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    // Top 24 bits → [0, 1)
    (h >> 40) as f32 / 16_777_216.0
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
    let mut rng = rand::rng();
    let placeholder = SpriteInstanceTransform::zeroed();

    for chunk in to_generate {
        // Sample regional density: normalise Perlin's ±0.866 output to [0, 1].
        let raw = perlin.sample(
            chunk.x as f32 * CHUNK_NOISE_FREQ,
            chunk.y as f32 * CHUNK_NOISE_FREQ,
            chunk.z as f32 * CHUNK_NOISE_FREQ,
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
        let entity = commands.push((
            AsteroidMarker,
            AsteroidChainId(chain_id),
            SpriteId { model_id: slot },
            NewtonBody {
                mass: 1.0,
                pos: chunk_centre,
                vel: DVec3::ZERO,
                orientation: DQuat::IDENTITY,
                angular_vel,
            },
        ));
        loaded.0.insert(entity);
    }
}

/// Update the slot↔entity maps after a GPU swap-remove moved `displaced_old` → `current_slot`.
/// Returns the entity whose `SpriteId` needs updating to `current_slot`, if any.
fn apply_swap_remove(
    current_slot: u32,
    displaced_old: u32,
    slot_to_entity: &mut HashMap<u32, Entity>,
    entity_to_slot: &mut HashMap<Entity, u32>,
) -> Option<Entity> {
    let displaced_entity = slot_to_entity.remove(&displaced_old)?;
    entity_to_slot.insert(displaced_entity, current_slot);
    slot_to_entity.insert(current_slot, displaced_entity);
    Some(displaced_entity)
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
    let mut slot_to_entity: HashMap<u32, Entity> = HashMap::new();
    let mut entity_to_slot: HashMap<Entity, u32> = HashMap::new();

    for &entity in &loaded.0 {
        let Ok(entry) = world.entry_ref(entity) else {
            continue;
        };
        let Ok(body) = entry.get_component::<NewtonBody>() else {
            continue;
        };
        let chunk = world_to_chunk(body.pos);
        let d = chunk - center;

        let Ok(sprite) = entry.get_component::<SpriteId>() else {
            continue;
        };
        slot_to_entity.insert(sprite.model_id, entity);
        entity_to_slot.insert(entity, sprite.model_id);
        if d.dot(d) > r2 {
            to_unload.push((entity, chunk));
        }
    }

    for (entity, chunk) in to_unload {
        let current_slot = match entity_to_slot.remove(&entity) {
            Some(s) => s,
            None => continue,
        };
        slot_to_entity.remove(&current_slot);

        if let Some(displaced_old) = gpu.remove_sprite_instance(current_slot as usize) {
            if let Some(displaced_entity) = apply_swap_remove(
                current_slot,
                displaced_old as u32,
                &mut slot_to_entity,
                &mut entity_to_slot,
            ) {
                if let Ok(mut entry) = world.entry_mut(displaced_entity) {
                    if let Ok(sprite) = entry.get_component_mut::<SpriteId>() {
                        sprite.model_id = current_slot;
                    }
                }
            }
        }

        if let Ok(entry) = world.entry_ref(entity) {
            if let Ok(chain) = entry.get_component::<AsteroidChainId>() {
                gpu.remove_sprite_model(chain.0);
            }
        }

        visited.0.remove(&chunk);
        loaded.0.remove(&entity);
        commands.remove(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_swap_remove, chunk_spawn_hash};
    use glam::IVec3;
    use legion::Entity;
    use std::collections::HashMap;

    fn make_entities(n: usize) -> (legion::World, Vec<Entity>) {
        let mut world = legion::World::default();
        let entities: Vec<Entity> = (0..n).map(|_| world.push((0u8,))).collect();
        (world, entities)
    }

    // ── chunk_spawn_hash ──────────────────────────────────────────────────────

    #[test]
    fn chunk_hash_deterministic() {
        let chunk = IVec3::new(1, 2, 3);
        assert_eq!(chunk_spawn_hash(42, chunk), chunk_spawn_hash(42, chunk));
    }

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

    // ── apply_swap_remove ─────────────────────────────────────────────────────

    #[test]
    fn swap_remove_missing_displaced_returns_none() {
        let (_world, entities) = make_entities(1);
        let e0 = entities[0];
        let mut s2e: HashMap<u32, Entity> = HashMap::from([(0, e0)]);
        let mut e2s: HashMap<Entity, u32> = HashMap::from([(e0, 0)]);
        // displaced_old=99 is absent — nothing to reassign.
        assert!(apply_swap_remove(0, 99, &mut s2e, &mut e2s).is_none());
    }

    #[test]
    fn swap_remove_normal_swap() {
        // Entities at slots 0, 1, 2. Remove slot 0; GPU moves e2 (slot 2) → slot 0.
        let (_world, entities) = make_entities(3);
        let [e0, e1, e2] = [entities[0], entities[1], entities[2]];
        let mut s2e = HashMap::from([(0u32, e0), (1, e1), (2, e2)]);
        let mut e2s = HashMap::from([(e0, 0u32), (e1, 1), (e2, 2)]);

        // Caller removes the departing entity before calling apply_swap_remove.
        s2e.remove(&0);
        e2s.remove(&e0);

        let result = apply_swap_remove(0, 2, &mut s2e, &mut e2s);

        assert_eq!(result, Some(e2));
        assert_eq!(s2e[&0], e2, "slot 0 now maps to e2");
        assert_eq!(e2s[&e2], 0, "e2 now tracks slot 0");
        assert!(!s2e.contains_key(&2), "old slot 2 must be vacated");
        assert_eq!(s2e[&1], e1, "e1 at slot 1 is untouched");
    }

    #[test]
    fn swap_remove_sequential_unloads() {
        // 4 entities. Unload slot 0 (e3 swaps 3→0), then unload slot 1 (e2 swaps 2→1).
        let (_world, entities) = make_entities(4);
        let [e0, e1, e2, e3] = [entities[0], entities[1], entities[2], entities[3]];
        let mut s2e = HashMap::from([(0u32, e0), (1, e1), (2, e2), (3, e3)]);
        let mut e2s = HashMap::from([(e0, 0u32), (e1, 1), (e2, 2), (e3, 3)]);

        // First unload: e0 at slot 0; GPU moves e3 (slot 3) → slot 0.
        s2e.remove(&0);
        e2s.remove(&e0);
        assert_eq!(apply_swap_remove(0, 3, &mut s2e, &mut e2s), Some(e3));

        // Second unload: e1 at slot 1; GPU moves e2 (slot 2) → slot 1.
        s2e.remove(&1);
        e2s.remove(&e1);
        assert_eq!(apply_swap_remove(1, 2, &mut s2e, &mut e2s), Some(e2));

        assert_eq!(e2s[&e3], 0);
        assert_eq!(e2s[&e2], 1);
        assert_eq!(s2e[&0], e3);
        assert_eq!(s2e[&1], e2);
        assert!(!s2e.contains_key(&2), "slot 2 must be vacated");
        assert!(!s2e.contains_key(&3), "slot 3 must be vacated");
    }
}
