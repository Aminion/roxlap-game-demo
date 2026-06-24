use std::collections::HashMap;

use legion::{systems::CommandBuffer, world::SubWorld, Entity, EntityStore, *};
use roxlap_gpu::GpuRenderer;

use crate::components::sprite_id::Sprite;

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

#[cfg(test)]
mod tests {
    use super::{apply_swap_remove, SpriteMaps};
    use legion::Entity;
    use std::collections::HashMap;

    fn make_entities(n: usize) -> (legion::World, Vec<Entity>) {
        let mut world = legion::World::default();
        let entities: Vec<Entity> = (0..n).map(|_| world.push((0u8,))).collect();
        (world, entities)
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
