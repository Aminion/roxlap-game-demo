use std::collections::HashMap;

use bytemuck::Zeroable;
use glam::{DQuat, DVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, EntityStore, *};
use rand::RngExt;
use roxlap_gpu::{GpuRenderer, SpriteInstance, SpriteInstanceTransform};

use crate::{
    components::{
        asteroid::{AsteroidChainId, AsteroidMarker, AsteroidModel},
        miner::Miner,
        newton_body::NewtonBody,
        presence_position::PresencePosition,
        sprite_id::SpriteId,
    },
    generation::chunks::{missing_chunks, world_to_chunk, CHUNK_SIZE, LOAD_RADIUS},
    world::build_asteroid_sprite_model,
    LoadedAsteroids, SpriteData, VisitedChunks,
};

const ASTEROIDS_PER_CHUNK: u32 = 1;
const UPDATE_DIST_SQ: f64 = (CHUNK_SIZE as f64 / 2.0) * (CHUNK_SIZE as f64 / 2.0);

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[read_component(AsteroidChainId)]
#[read_component(AsteroidModel)]
#[write_component(PresencePosition)]
#[write_component(SpriteId)]
pub fn presence_position_update(
    #[resource] visited: &mut VisitedChunks,
    #[resource] loaded: &mut LoadedAsteroids,
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

    if let Some(ship_pos) = updated_pos {
        update_sprites(ship_pos, loaded, gpu, sprite_data, world, commands);
        populate_chunks(ship_pos, visited, loaded, gpu, sprite_data, commands);
    }
}

fn populate_chunks(
    ship_pos: DVec3,
    visited: &mut VisitedChunks,
    loaded: &mut LoadedAsteroids,
    gpu: &mut GpuRenderer,
    sprite_data: &mut SpriteData,
    commands: &mut CommandBuffer,
) {
    let to_generate: Vec<_> = missing_chunks(ship_pos, LOAD_RADIUS, &visited.0).collect();

    if to_generate.is_empty() {
        return;
    }

    let mut rng = rand::rng();
    let placeholder = SpriteInstanceTransform::zeroed();

    for chunk in to_generate {
        let chunk_centre = (chunk.as_dvec3() + DVec3::splat(0.5)) * CHUNK_SIZE as f64;
        for _ in 0..ASTEROIDS_PER_CHUNK {
            let model = build_asteroid_sprite_model();
            let chain_id = sprite_data.registry.add(model.clone());
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
                AsteroidModel(model),
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
        visited.0.insert(chunk);
    }
}

/// Single pass over all loaded asteroids: deactivate those that left the presence radius,
/// activate those that entered it.
fn update_sprites(
    ship_pos: DVec3,
    loaded: &LoadedAsteroids,
    gpu: &mut GpuRenderer,
    sprite_data: &mut SpriteData,
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
) {
    use roxlap_gpu::SpriteModel;

    let center = world_to_chunk(ship_pos);
    let r2 = LOAD_RADIUS * LOAD_RADIUS;
    let placeholder = SpriteInstanceTransform::zeroed();

    let mut to_deactivate: Vec<Entity> = Vec::new();
    let mut to_activate: Vec<(Entity, SpriteModel)> = Vec::new();
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
        let in_range = d.dot(d) <= r2;

        if let Ok(sprite) = entry.get_component::<SpriteId>() {
            slot_to_entity.insert(sprite.model_id, entity);
            entity_to_slot.insert(entity, sprite.model_id);
            if !in_range {
                to_deactivate.push(entity);
            }
        } else if in_range {
            let Ok(asteroid_model) = entry.get_component::<AsteroidModel>() else {
                continue;
            };
            to_activate.push((entity, asteroid_model.0.clone()));
        }
    }

    if !to_deactivate.is_empty() {
        for entity in to_deactivate {
            let current_slot = match entity_to_slot.remove(&entity) {
                Some(s) => s,
                None => continue,
            };
            slot_to_entity.remove(&current_slot);

            if let Some(displaced_old) = gpu.remove_sprite_instance(current_slot as usize) {
                if let Some(displaced_entity) = slot_to_entity.remove(&(displaced_old as u32)) {
                    entity_to_slot.insert(displaced_entity, current_slot);
                    slot_to_entity.insert(current_slot, displaced_entity);
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

            commands.remove_component::<SpriteId>(entity);
        }
    }

    if !to_activate.is_empty() {
        for (entity, model_clone) in to_activate {
            let new_chain_id = sprite_data.registry.add(model_clone);
            gpu.add_sprite_model(&sprite_data.registry, new_chain_id);
            let slot = gpu.append_sprite_instances(
                &sprite_data.registry,
                &[SpriteInstance {
                    model_id: new_chain_id,
                    transform: placeholder,
                }],
            );
            commands.add_component(entity, AsteroidChainId(new_chain_id));
            commands.add_component(entity, SpriteId { model_id: slot });
        }
    }
}
