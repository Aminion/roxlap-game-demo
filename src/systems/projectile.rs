use std::collections::HashMap;

use glam::DVec3;
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, *};
use roxlap_gpu::GpuRenderer;

use crate::{
    components::{
        aabb::Aabb, asteroid::AsteroidChainId, newton_body::NewtonBody, projectile::Projectile,
        sprite_id::SpriteId,
    },
    generation::chunks::world_to_chunk,
    systems::presence_position::perform_despawn,
    Dt, LoadedAsteroids, VisitedChunks,
};

#[system]
#[write_component(Projectile)]
#[read_component(NewtonBody)]
#[read_component(Aabb)]
#[read_component(AsteroidChainId)]
#[write_component(SpriteId)]
pub fn projectile(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] dt: &Dt,
    #[resource] gpu: &mut GpuRenderer,
    #[resource] loaded: &mut LoadedAsteroids,
    #[resource] visited: &mut VisitedChunks,
) {
    // Collect projectile states and tick lifetimes.
    struct ProjState {
        entity: Entity,
        pos: DVec3,
        lifetime: f64,
        chain_id: u32,
        slot: u32,
    }
    let mut projectiles: Vec<ProjState> = Vec::new();
    {
        let mut q = <(Entity, &mut Projectile, &NewtonBody, &SpriteId)>::query();
        for (entity, proj, body, sprite) in q.iter_mut(world) {
            proj.lifetime -= dt.0;
            projectiles.push(ProjState {
                entity: *entity,
                pos: body.pos,
                lifetime: proj.lifetime,
                chain_id: proj.chain_id,
                slot: sprite.model_id,
            });
        }
    }

    // Collect asteroid states.
    struct AstState {
        entity: Entity,
        pos: DVec3,
        half_extent: f32,
        chain_id: u32,
        slot: u32,
    }
    let mut asteroids: Vec<AstState> = Vec::new();
    for &entity in &loaded.0 {
        let Ok(entry) = world.entry_ref(entity) else {
            continue;
        };
        let Ok(body) = entry.get_component::<NewtonBody>() else {
            continue;
        };
        let Ok(aabb) = entry.get_component::<Aabb>() else {
            continue;
        };
        let Ok(chain) = entry.get_component::<AsteroidChainId>() else {
            continue;
        };
        let Ok(sprite) = entry.get_component::<SpriteId>() else {
            continue;
        };
        asteroids.push(AstState {
            entity,
            pos: body.pos,
            half_extent: aabb.half_extent,
            chain_id: chain.0,
            slot: sprite.model_id,
        });
    }

    // Determine which projectiles expire or hit an asteroid.
    let mut proj_to_remove: Vec<(Entity, u32, u32)> = Vec::new(); // (entity, chain_id, slot)
    let mut ast_to_remove: Vec<Entity> = Vec::new();

    for p in &projectiles {
        if p.lifetime <= 0.0 {
            proj_to_remove.push((p.entity, p.chain_id, p.slot));
            continue;
        }
        for a in &asteroids {
            let h = (a.half_extent + 0.5) as f64;
            let d = p.pos - a.pos;
            if d.x.abs() <= h && d.y.abs() <= h && d.z.abs() <= h {
                proj_to_remove.push((p.entity, p.chain_id, p.slot));
                if !ast_to_remove.contains(&a.entity) {
                    ast_to_remove.push(a.entity);
                }
                break;
            }
        }
    }

    if proj_to_remove.is_empty() && ast_to_remove.is_empty() {
        return;
    }

    // Build a full slot↔entity map covering all sprite entities.
    let mut slot_to_entity: HashMap<u32, Entity> = HashMap::new();
    let mut entity_to_slot: HashMap<Entity, u32> = HashMap::new();
    for a in &asteroids {
        slot_to_entity.insert(a.slot, a.entity);
        entity_to_slot.insert(a.entity, a.slot);
    }
    for p in &projectiles {
        slot_to_entity.insert(p.slot, p.entity);
        entity_to_slot.insert(p.entity, p.slot);
    }

    // Despawn expired/hit projectiles.
    for (proj_entity, chain_id, _) in &proj_to_remove {
        let Some(current_slot) = entity_to_slot.remove(proj_entity) else {
            continue;
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

        gpu.remove_sprite_model(*chain_id);
        commands.remove(*proj_entity);
    }

    // Despawn hit asteroids.
    for ast_entity in ast_to_remove {
        let (chunk, chain_id) = {
            let Ok(entry) = world.entry_ref(ast_entity) else {
                continue;
            };
            let Ok(body) = entry.get_component::<NewtonBody>() else {
                continue;
            };
            let Ok(chain) = entry.get_component::<AsteroidChainId>() else {
                continue;
            };
            (world_to_chunk(body.pos), chain.0)
        };

        perform_despawn(
            ast_entity,
            chunk,
            chain_id,
            &mut slot_to_entity,
            &mut entity_to_slot,
            world,
            commands,
            gpu,
            loaded,
            visited,
        );
    }
}
