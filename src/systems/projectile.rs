use std::collections::HashMap;

use glam::{DQuat, DVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, *};
use roxlap_gpu::{GpuRenderer, SpriteModel};

use crate::{
    components::{
        aabb::Aabb, asteroid::AsteroidChainId, newton_body::NewtonBody, projectile::Projectile,
        sprite_id::SpriteId,
    },
    systems::presence_position::perform_despawn,
    Dt, LoadedAsteroids, SpriteData,
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
    #[resource] sprite_data: &SpriteData,
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
        orientation: DQuat,
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
            orientation: body.orientation,
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
            let hit_voxel = if d.x.abs() <= h && d.y.abs() <= h && d.z.abs() <= h {
                voxel_hit(p.pos, a.pos, a.orientation, sprite_data.registry.model(a.chain_id))
            } else {
                None
            };
            if hit_voxel.is_some() {
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
        let chain_id = {
            let Ok(entry) = world.entry_ref(ast_entity) else {
                continue;
            };
            let Ok(chain) = entry.get_component::<AsteroidChainId>() else {
                continue;
            };
            chain.0
        };

        perform_despawn(
            ast_entity,
            chain_id,
            &mut slot_to_entity,
            &mut entity_to_slot,
            world,
            commands,
            gpu,
            loaded,
        );
    }
}

/// Returns the model-local voxel coordinates `(x, y, z)` of the hit, or `None`.
///
/// Point test only — a fast-moving projectile may tunnel through thin geometry
/// between frames if it skips more than one voxel per frame.
fn voxel_hit(
    proj_pos: DVec3,
    ast_pos: DVec3,
    ast_orientation: DQuat,
    model: &SpriteModel,
) -> Option<(u32, u32, u32)> {
    let local = ast_orientation.inverse() * (proj_pos - ast_pos);
    let vx = (local.x / model.voxel_world_size as f64 + model.pivot[0] as f64).floor() as i64;
    let vy = (local.y / model.voxel_world_size as f64 + model.pivot[1] as f64).floor() as i64;
    let vz = (local.z / model.voxel_world_size as f64 + model.pivot[2] as f64).floor() as i64;
    if vx < 0
        || vy < 0
        || vz < 0
        || vx >= model.dims[0] as i64
        || vy >= model.dims[1] as i64
        || vz >= model.dims[2] as i64
    {
        return None;
    }
    let (vx, vy, vz) = (vx as u32, vy as u32, vz as u32);
    let col = (vx + vy * model.dims[0]) as usize;
    let base = col * model.occ_words_per_col as usize;
    let occupied = (model.occupancy[base + vz as usize / 32] >> (vz % 32)) & 1 == 1;
    occupied.then_some((vx, vy, vz))
}

#[cfg(test)]
mod tests {
    use super::voxel_hit;
    use glam::{DQuat, DVec3};
    use roxlap_gpu::SpriteModel;
    use std::f64::consts::FRAC_PI_2;

    /// 3×1×1 model with only voxel (2, 0, 0) occupied.
    /// pivot = (1.5, 0.5, 0.5) — geometric centre.
    fn make_3x1x1() -> SpriteModel {
        // column = x + y*mx; 3 columns, 1 occ word each
        let mut occupancy = vec![0u32; 3];
        occupancy[2] = 1u32; // col 2, bit 0 → voxel (2, 0, 0)
        SpriteModel {
            dims: [3, 1, 1],
            occ_words_per_col: 1,
            pivot: [1.5, 0.5, 0.5],
            occupancy,
            colors: vec![0xFF_FF_FF_FF],
            dirs: vec![0],
            color_offsets: vec![0, 0, 0, 1],
            voxel_world_size: 1.0,
        }
    }

    #[test]
    fn hit_occupied_voxel_identity() {
        // Voxel (2,0,0) center in model space = (2.5, 0.5, 0.5).
        // World offset from pivot = (2.5-1.5, 0, 0) = (1, 0, 0).
        assert_eq!(
            voxel_hit(DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO, DQuat::IDENTITY, &make_3x1x1()),
            Some((2, 0, 0))
        );
    }

    #[test]
    fn miss_empty_voxel_identity() {
        // Voxel (0,0,0) is empty. Its world offset from pivot = (0.5-1.5, 0, 0) = (-1, 0, 0).
        assert_eq!(
            voxel_hit(DVec3::new(-1.0, 0.0, 0.0), DVec3::ZERO, DQuat::IDENTITY, &make_3x1x1()),
            None
        );
    }

    #[test]
    fn hit_rotated_90_degrees_y() {
        // rotation_y(π/2) maps model +X → world -Z.
        // Voxel (2,0,0) body-local offset from pivot = (1, 0, 0).
        // World offset = rotation_y(π/2) * (1,0,0) ≈ (0, 0, -1).
        let rot = DQuat::from_rotation_y(FRAC_PI_2);
        assert_eq!(
            voxel_hit(DVec3::new(0.0, 0.0, -1.0), DVec3::ZERO, rot, &make_3x1x1()),
            Some((2, 0, 0))
        );
    }

    #[test]
    fn miss_wrong_axis_after_rotation() {
        // If we incorrectly applied `orientation * ...` instead of `inverse() * ...`,
        // (1,0,0) in world would map into the occupied voxel. With the correct
        // `inverse()`, (1,0,0) lands in z=1 which is out of bounds.
        let rot = DQuat::from_rotation_y(FRAC_PI_2);
        assert_eq!(
            voxel_hit(DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO, rot, &make_3x1x1()),
            None
        );
    }

    #[test]
    fn miss_out_of_bounds() {
        assert_eq!(
            voxel_hit(DVec3::new(10.0, 0.0, 0.0), DVec3::ZERO, DQuat::IDENTITY, &make_3x1x1()),
            None
        );
    }
}
