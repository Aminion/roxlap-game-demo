use std::collections::HashMap;

use glam::DVec3;
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, *};
use roxlap_core::ray_aabb::clip_ray_to_aabb;

use crate::{
    components::{
        aabb::Aabb, asteroid::AsteroidChainId, canon::Canon, miner::Miner, newton_body::NewtonBody,
        sprite_id::SpriteId,
    },
    generation::chunks::world_to_chunk,
    systems::presence_position::perform_despawn,
    LoadedAsteroids, VisitedChunks,
};
use roxlap_gpu::GpuRenderer;

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[read_component(Aabb)]
#[read_component(AsteroidChainId)]
#[write_component(Canon)]
#[write_component(SpriteId)]
pub fn shooting(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] gpu: &mut GpuRenderer,
    #[resource] loaded: &mut LoadedAsteroids,
    #[resource] visited: &mut VisitedChunks,
) {
    // Check fire state and get ray from the miner's Canon + NewtonBody.
    let (ray_origin, ray_dir) = {
        let mut miner_q = <(&Miner, &NewtonBody, &mut Canon)>::query();
        let Some((_, body, canon)) = miner_q.iter_mut(world).next() else {
            return;
        };
        if !canon.firing || canon.cooldown > 0.0 {
            return;
        }
        let ray_origin = body.pos;
        let ray_dir = (body.orientation * DVec3::NEG_Z).normalize();
        canon.cooldown = 0.5;
        (ray_origin, ray_dir)
    };

    // Build slot↔entity maps and find the closest AABB hit.
    let mut slot_to_entity: HashMap<u32, Entity> = HashMap::new();
    let mut entity_to_slot: HashMap<Entity, u32> = HashMap::new();
    let mut best: Option<(f32, Entity)> = None;

    for &entity in &loaded.0 {
        let Ok(entry) = world.entry_ref(entity) else {
            continue;
        };
        let Ok(ab_body) = entry.get_component::<NewtonBody>() else {
            continue;
        };
        let Ok(aabb) = entry.get_component::<Aabb>() else {
            continue;
        };
        let Ok(sprite) = entry.get_component::<SpriteId>() else {
            continue;
        };
        slot_to_entity.insert(sprite.model_id, entity);
        entity_to_slot.insert(entity, sprite.model_id);

        // Translate AABB to ray-origin-relative coordinates to avoid f32 precision loss.
        let rel = (ab_body.pos - ray_origin).as_vec3();
        let h = aabb.half_extent;
        let aabb_min = [rel.x - h, rel.y - h, rel.z - h];
        let aabb_max = [rel.x + h, rel.y + h, rel.z + h];
        let dir_f32 = ray_dir.as_vec3().to_array();

        if let Some((t_enter, t_exit)) =
            clip_ray_to_aabb([0.0, 0.0, 0.0], dir_f32, aabb_min, aabb_max)
        {
            // t_exit >= 0 guaranteed by clip_ray_to_aabb; use max(t_enter, 0) as sort key.
            let t = t_enter.max(0.0);
            if t_exit > 0.0 && best.map_or(true, |(bt, _)| t < bt) {
                best = Some((t, entity));
            }
        }
    }

    let Some((_, hit_entity)) = best else {
        return;
    };

    // Fetch the data we need before consuming the entity.
    let (chunk, chain_id) = {
        let Ok(entry) = world.entry_ref(hit_entity) else {
            return;
        };
        let Ok(body) = entry.get_component::<NewtonBody>() else {
            return;
        };
        let Ok(chain) = entry.get_component::<AsteroidChainId>() else {
            return;
        };
        (world_to_chunk(body.pos), chain.0)
    };

    perform_despawn(
        hit_entity,
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
