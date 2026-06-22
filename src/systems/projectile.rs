use bytemuck::Zeroable;
use glam::{DQuat, DVec3, IVec3, UVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, *};
use rand::RngExt;
use roxlap_gpu::{GpuRenderer, SpriteInstance, SpriteInstanceTransform, SpriteModel};

use crate::{
    components::{
        aabb::Aabb,
        asteroid::{AsteroidMinerals, AsteroidVoxelInfo, ChainId, CrystalMarker},
        newton_body::NewtonBody,
        projectile::Projectile,
        sprite_id::SpriteId,
    },
    systems::presence_position::{build_sprite_maps, perform_despawn},
    world::build_crystal_sprite_model,
    Dt, LoadedAsteroids, SpriteData,
};

/// Voxel radius of the crater carved on each hit.
const HIT_CARVE_RADIUS: u32 = 4;

/// Scales raw projectile momentum (mass × speed) into effective impulse.
/// Keeps the direction physics-correct while tuning the magnitude for
/// feel — without it a 0.001 kg bullet hitting a 1 kg asteroid barely
/// nudges it.
const HIT_IMPULSE_FACTOR: f64 = 5.0;

#[system]
#[write_component(Projectile)]
#[write_component(NewtonBody)]
#[read_component(Aabb)]
#[read_component(ChainId)]
#[read_component(AsteroidVoxelInfo)]
#[write_component(AsteroidMinerals)]
#[write_component(SpriteId)]
pub fn projectile(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] dt: &Dt,
    #[resource] gpu: &mut GpuRenderer,
    #[resource] loaded: &mut LoadedAsteroids,
    #[resource] sprite_data: &mut SpriteData,
) {
    // Collect projectile states and tick lifetimes.
    struct ProjState {
        entity: Entity,
        pos: DVec3,
        vel: DVec3,
        mass: f64,
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
                vel: body.vel,
                mass: body.mass,
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
        vel: DVec3,
        angular_vel: DVec3,
        orientation: DQuat,
        half_extent: f32,
        mass: f64,
        chain_id: u32,
        slot: u32,
        minerals: Vec<UVec3>,
        initial_voxel_count: u32,
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
        let Ok(chain) = entry.get_component::<ChainId>() else {
            continue;
        };
        let Ok(sprite) = entry.get_component::<SpriteId>() else {
            continue;
        };
        let minerals = entry
            .get_component::<AsteroidMinerals>()
            .map(|m| m.points.clone())
            .unwrap_or_default();
        let initial_voxel_count = entry
            .get_component::<AsteroidVoxelInfo>()
            .map(|v| v.initial_count)
            .unwrap_or(0);
        asteroids.push(AstState {
            entity,
            pos: body.pos,
            vel: body.vel,
            angular_vel: body.angular_vel,
            orientation: body.orientation,
            half_extent: aabb.half_extent,
            mass: body.mass,
            chain_id: chain.0,
            slot: sprite.model_id,
            minerals,
            initial_voxel_count,
        });
    }

    // Determine which projectiles expire or hit an asteroid.
    struct HitData {
        ast_entity: Entity,
        ast_chain_id: u32,
        ast_pos: DVec3,
        ast_vel: DVec3,
        ast_angular_vel: DVec3,
        ast_mass: f64,
        ast_orientation: DQuat,
        ast_half_extent: f32,
        hit_voxel: (u32, u32, u32),
        proj_vel: DVec3,
        proj_mass: f64,
        minerals: Vec<UVec3>,
        initial_voxel_count: u32,
    }
    let mut proj_to_remove: Vec<(Entity, u32, u32)> = Vec::new(); // (entity, chain_id, slot)
    let mut ast_hits: Vec<HitData> = Vec::new();

    for p in &projectiles {
        if p.lifetime <= 0.0 {
            proj_to_remove.push((p.entity, p.chain_id, p.slot));
            continue;
        }
        for a in &asteroids {
            let h = (a.half_extent + 0.5) as f64;
            let d = p.pos - a.pos;
            let hit_voxel = if d.x.abs() <= h && d.y.abs() <= h && d.z.abs() <= h {
                voxel_hit(
                    p.pos,
                    a.pos,
                    a.orientation,
                    sprite_data.registry.model(a.chain_id),
                )
            } else {
                None
            };
            if let Some(hit_voxel) = hit_voxel {
                proj_to_remove.push((p.entity, p.chain_id, p.slot));
                if !ast_hits.iter().any(|h| h.ast_entity == a.entity) {
                    ast_hits.push(HitData {
                        ast_entity: a.entity,
                        ast_chain_id: a.chain_id,
                        ast_pos: a.pos,
                        ast_vel: a.vel,
                        ast_angular_vel: a.angular_vel,
                        ast_mass: a.mass,
                        ast_orientation: a.orientation,
                        ast_half_extent: a.half_extent,
                        hit_voxel,
                        proj_vel: p.vel,
                        proj_mass: p.mass,
                        minerals: a.minerals.clone(),
                        initial_voxel_count: a.initial_voxel_count,
                    });
                }
                break;
            }
        }
    }

    if proj_to_remove.is_empty() && ast_hits.is_empty() {
        return;
    }

    // Build a full slot↔entity map from ALL sprite entities so that any
    // swap-remove (projectile, asteroid, or crystal displaced) is handled correctly.
    let mut maps = build_sprite_maps(world);

    // Despawn expired/hit projectiles.
    for (proj_entity, chain_id, _) in &proj_to_remove {
        let Some(current_slot) = maps.entity_to_slot.remove(proj_entity) else {
            continue;
        };
        maps.slot_to_entity.remove(&current_slot);

        if let Some(displaced_old) = gpu.remove_sprite_instance(current_slot as usize) {
            if let Some(displaced_entity) = maps.slot_to_entity.remove(&(displaced_old as u32)) {
                maps.entity_to_slot.insert(displaced_entity, current_slot);
                maps.slot_to_entity.insert(current_slot, displaced_entity);
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

    // Process hit asteroids: carve a sphere, apply physics impulse, despawn if empty.
    for hit in ast_hits {
        let (vx, vy, vz) = hit.hit_voxel;
        let (pivot, dims) = {
            let m = sprite_data.registry.model(hit.ast_chain_id);
            (m.pivot, m.dims)
        };

        // Find mineral points inside the carved sphere before we destroy them.
        let carve_r = HIT_CARVE_RADIUS as i32;
        let hit_voxel_i = IVec3::new(vx as i32, vy as i32, vz as i32);
        let hit_minerals: Vec<UVec3> = hit
            .minerals
            .iter()
            .filter(|p| (p.as_ivec3() - hit_voxel_i).length_squared() <= carve_r * carve_r)
            .copied()
            .collect();

        // Carve a sphere of HIT_CARVE_RADIUS centred on the hit voxel.
        let (empty, current_voxel_count) = {
            let model = sprite_data.registry.model_mut(hit.ast_chain_id);
            let r = HIT_CARVE_RADIUS as i32;
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx * dx + dy * dy + dz * dz > r * r {
                            continue;
                        }
                        let (cx, cy, cz) = (vx as i32 + dx, vy as i32 + dy, vz as i32 + dz);
                        if cx >= 0
                            && cy >= 0
                            && cz >= 0
                            && cx < dims[0] as i32
                            && cy < dims[1] as i32
                            && cz < dims[2] as i32
                        {
                            model.set_voxel(cx as u32, cy as u32, cz as u32, None);
                        }
                    }
                }
            }
            (model.colors.is_empty(), model.colors.len() as u32)
        };

        // Trigger full destruction when fewer than 20 % of the original voxels remain.
        let force_destroy = !empty && current_voxel_count * 5 < hit.initial_voxel_count;
        let destroy = empty || force_destroy;

        // On a normal carve only hit_minerals spawn; on full destruction all remaining
        // mineral points do (so the killing blow never silently swallows crystals).
        let crystals_to_spawn: &[UVec3] = if destroy {
            &hit.minerals
        } else {
            &hit_minerals
        };

        let mut rng = rand::rng();
        let pivot_vec = DVec3::new(pivot[0] as f64, pivot[1] as f64, pivot[2] as f64);
        for &p in crystals_to_spawn {
            let local = p.as_dvec3() + DVec3::splat(0.5) - pivot_vec;
            let crystal_world = hit.ast_pos + hit.ast_orientation * local;

            let spin = DVec3::new(
                (rng.random::<f64>() - 0.5) * 4.0,
                (rng.random::<f64>() - 0.5) * 4.0,
                (rng.random::<f64>() - 0.5) * 4.0,
            );
            let eject_dir = (crystal_world - hit.ast_pos).normalize_or_zero();
            let eject_speed = rng.random_range(0.5f64..2.0);
            let crystal_vel = hit.ast_vel + eject_dir * eject_speed;

            let c_chain = sprite_data.registry.add(build_crystal_sprite_model());
            gpu.add_sprite_model(&sprite_data.registry, c_chain);
            let c_slot = gpu.append_sprite_instances(
                &sprite_data.registry,
                &[SpriteInstance {
                    model_id: c_chain,
                    transform: SpriteInstanceTransform::zeroed(),
                }],
            );
            commands.push((
                CrystalMarker,
                ChainId(c_chain),
                NewtonBody {
                    mass: 0.01,
                    pos: crystal_world,
                    vel: crystal_vel,
                    orientation: DQuat::IDENTITY,
                    angular_vel: spin,
                },
                SpriteId { model_id: c_slot },
                Aabb { half_extent: 1.5 },
            ));
        }

        // Re-upload the edited model to the GPU.
        gpu.update_sprite_model(&sprite_data.registry, hit.ast_chain_id);

        if destroy {
            perform_despawn(
                hit.ast_entity,
                hit.ast_chain_id,
                &mut maps,
                world,
                commands,
                gpu,
                loaded,
            );
        } else {
            // Apply linear and angular impulse from the projectile hit.
            //
            // effective_impulse = proj_mass × proj_vel × HIT_IMPULSE_FACTOR
            // delta_vel         = effective_impulse / ast_mass
            // lever             = world-space vector from asteroid centre to hit voxel
            // delta_omega       = lever × effective_impulse / moment_of_inertia
            //   (moment of inertia for a solid sphere ≈ 2/5 × mass × radius²)
            let hit_local = DVec3::new(
                vx as f64 + 0.5 - pivot[0] as f64,
                vy as f64 + 0.5 - pivot[1] as f64,
                vz as f64 + 0.5 - pivot[2] as f64,
            );
            let lever = hit.ast_orientation * hit_local; // world space
            let effective_impulse = hit.proj_vel * hit.proj_mass * HIT_IMPULSE_FACTOR;
            let delta_vel = effective_impulse / hit.ast_mass;
            let radius = hit.ast_half_extent as f64;
            let moment = 0.4 * hit.ast_mass * radius * radius;
            let delta_omega = lever.cross(effective_impulse) / moment;

            if let Ok(mut entry) = world.entry_mut(hit.ast_entity) {
                if let Ok(body) = entry.get_component_mut::<NewtonBody>() {
                    body.vel += delta_vel;
                    body.angular_vel += delta_omega;
                }
                if !hit_minerals.is_empty() {
                    if let Ok(minerals) = entry.get_component_mut::<AsteroidMinerals>() {
                        minerals.points.retain(|p| !hit_minerals.contains(p));
                    }
                }
            }
        }
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
            voxel_hit(
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::ZERO,
                DQuat::IDENTITY,
                &make_3x1x1()
            ),
            Some((2, 0, 0))
        );
    }

    #[test]
    fn miss_empty_voxel_identity() {
        // Voxel (0,0,0) is empty. Its world offset from pivot = (0.5-1.5, 0, 0) = (-1, 0, 0).
        assert_eq!(
            voxel_hit(
                DVec3::new(-1.0, 0.0, 0.0),
                DVec3::ZERO,
                DQuat::IDENTITY,
                &make_3x1x1()
            ),
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
            voxel_hit(
                DVec3::new(10.0, 0.0, 0.0),
                DVec3::ZERO,
                DQuat::IDENTITY,
                &make_3x1x1()
            ),
            None
        );
    }
}
