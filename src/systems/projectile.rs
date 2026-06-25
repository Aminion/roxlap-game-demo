use glam::{DQuat, DVec3, IVec3, UVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, Entity, *};
use rand::RngExt;
use roxlap_gpu::SpriteModel;
use roxlap_render::{SceneRenderer, SpriteModelId};

use crate::{
    components::{
        aabb::Aabb,
        asteroid::{AsteroidMinerals, AsteroidVoxelInfo},
        crystal::CrystalMarker,
        newton_body::NewtonBody,
        projectile::Projectile,
        sprite_id::Sprite,
    },
    systems::sprite::perform_despawn,
    world::{build_crystal_sprite_model, spawn_sprite, sprite_model_to_kv6},
    Dt, LoadedAsteroids, SpriteData,
};

/// Voxel radius of the crater carved on each hit.
const HIT_CARVE_RADIUS: u32 = 4;

/// Scales raw projectile momentum (mass × speed) into effective impulse.
/// Keeps the direction physics-correct while tuning the magnitude for
/// feel — without it a 0.001 kg bullet hitting a 1 kg asteroid barely
/// nudges it.
const HIT_IMPULSE_FACTOR: f64 = 5.0;
/// Moment of inertia coefficient for a uniform solid sphere: I = (2/5)·m·r².
const SOLID_SPHERE_INERTIA: f64 = 0.4;

#[system]
#[write_component(Projectile)]
#[write_component(NewtonBody)]
#[read_component(Aabb)]
#[read_component(AsteroidVoxelInfo)]
#[write_component(AsteroidMinerals)]
#[write_component(Sprite)]
pub fn projectile(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] dt: &Dt,
    #[resource] renderer: &mut SceneRenderer,
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
    }
    let mut projectiles: Vec<ProjState> = Vec::new();
    {
        let mut q = <(Entity, &mut Projectile, &NewtonBody)>::query();
        for (entity, proj, body) in q.iter_mut(world) {
            proj.lifetime -= dt.0;
            projectiles.push(ProjState {
                entity: *entity,
                pos: body.pos,
                vel: body.vel,
                mass: body.mass,
                lifetime: proj.lifetime,
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
        aabb_min: DVec3,
        aabb_max: DVec3,
        radius: f64,
        mass: f64,
        chain_id: u32,
        model_id: SpriteModelId,
        initial_voxel_count: u32,
    }
    let mut asteroids: Vec<AstState> = Vec::with_capacity(loaded.0.len());
    for &entity in &loaded.0 {
        let entry = world
            .entry_ref(entity)
            .expect("loaded asteroid entity missing");
        let body = entry
            .get_component::<NewtonBody>()
            .expect("loaded asteroid missing NewtonBody");
        let aabb = entry
            .get_component::<Aabb>()
            .expect("loaded asteroid missing Aabb");
        let sprite = entry
            .get_component::<Sprite>()
            .expect("loaded asteroid missing Sprite");
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
            aabb_min: aabb.min,
            aabb_max: aabb.max,
            radius: ((aabb.max - aabb.min) * 0.5).max_element(),
            mass: body.mass,
            chain_id: sprite.chain_id,
            model_id: sprite.model_id,
            initial_voxel_count,
        });
    }

    // Determine which projectiles expire or hit an asteroid.
    struct HitData {
        ast_entity: Entity,
        ast_chain_id: u32,
        ast_model_id: SpriteModelId,
        ast_pos: DVec3,
        ast_vel: DVec3,
        ast_angular_vel: DVec3,
        ast_mass: f64,
        ast_orientation: DQuat,
        ast_radius: f64,
        hit_voxel: UVec3,
        proj_vel: DVec3,
        proj_mass: f64,
        initial_voxel_count: u32,
    }
    let mut proj_to_remove: Vec<Entity> = Vec::with_capacity(projectiles.len());
    let mut ast_hits: Vec<HitData> = Vec::new();

    for p in &projectiles {
        if p.lifetime <= 0.0 {
            proj_to_remove.push(p.entity);
            continue;
        }
        for a in &asteroids {
            let expanded_min = a.aabb_min - DVec3::splat(0.5);
            let expanded_max = a.aabb_max + DVec3::splat(0.5);
            let hit_voxel = if p.pos.cmpge(expanded_min).all() && p.pos.cmple(expanded_max).all() {
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
                proj_to_remove.push(p.entity);
                if !ast_hits.iter().any(|h| h.ast_entity == a.entity) {
                    ast_hits.push(HitData {
                        ast_entity: a.entity,
                        ast_chain_id: a.chain_id,
                        ast_model_id: a.model_id,
                        ast_pos: a.pos,
                        ast_vel: a.vel,
                        ast_angular_vel: a.angular_vel,
                        ast_mass: a.mass,
                        ast_orientation: a.orientation,
                        ast_radius: a.radius,
                        hit_voxel,
                        proj_vel: p.vel,
                        proj_mass: p.mass,
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

    // Despawn expired/hit projectiles.
    for proj_entity in &proj_to_remove {
        perform_despawn(*proj_entity, world, commands, renderer);
    }

    // Crystal spawn data collected during hit processing; spawned after all despawns so
    // their GPU slots aren't displaced by subsequent swap-removes in the same batch.
    struct PendingCrystal {
        pos: DVec3,
        vel: DVec3,
        spin: DVec3,
    }
    let mut pending_crystals: Vec<PendingCrystal> = Vec::new();

    // Process hit asteroids: carve a sphere, apply physics impulse, despawn if empty.
    for hit in ast_hits {
        let hv = hit.hit_voxel;
        let pivot = sprite_data.registry.model(hit.ast_chain_id).pivot;
        let pivot_vec = DVec3::from(pivot.map(|p| p as f64));

        // Read minerals from the world only for asteroids that were actually hit.
        let (hit_minerals, all_minerals) = match world.entry_ref(hit.ast_entity) {
            Ok(entry) => match entry.get_component::<AsteroidMinerals>() {
                Ok(m) => {
                    let all = m.points.clone();
                    let hit_m = minerals_in_radius(&all, hv.as_ivec3(), HIT_CARVE_RADIUS);
                    (hit_m, all)
                }
                Err(_) => (vec![], vec![]),
            },
            Err(_) => (vec![], vec![]),
        };

        let (empty, current_voxel_count) = carve_sphere(
            sprite_data.registry.model_mut(hit.ast_chain_id),
            hv.as_ivec3(),
            HIT_CARVE_RADIUS,
        );

        // Trigger full destruction when fewer than 20 % of the original voxels remain.
        let force_destroy = !empty && current_voxel_count * 5 < hit.initial_voxel_count;
        let destroy = empty || force_destroy;

        // On a normal carve only hit_minerals spawn; on full destruction all remaining
        // mineral points do (so the killing blow never silently swallows crystals).
        let crystals_to_spawn: &[UVec3] = if destroy {
            &all_minerals
        } else {
            &hit_minerals
        };

        let mut rng = rand::rng();
        for &p in crystals_to_spawn {
            let local = p.as_dvec3() + DVec3::splat(0.5) - pivot_vec;
            let crystal_world = hit.ast_pos + hit.ast_orientation * local;
            let spin = DVec3::new(
                rng.random_range(-2.0f64..2.0),
                rng.random_range(-2.0f64..2.0),
                rng.random_range(-2.0f64..2.0),
            );
            let eject_dir = (crystal_world - hit.ast_pos).normalize_or_zero();
            let eject_speed = rng.random_range(0.5f64..2.0);
            pending_crystals.push(PendingCrystal {
                pos: crystal_world,
                vel: hit.ast_vel + eject_dir * eject_speed,
                spin,
            });
        }

        // Re-upload the carved model to the renderer.
        renderer.refresh_sprite_model(
            hit.ast_model_id,
            &sprite_model_to_kv6(sprite_data.registry.model(hit.ast_chain_id)),
        );

        if destroy {
            perform_despawn(hit.ast_entity, world, commands, renderer);
            loaded.0.remove(&hit.ast_entity);
        } else {
            let (delta_vel, delta_omega) = hit_impulse(
                hit.proj_vel,
                hit.proj_mass,
                hit.ast_mass,
                hit.ast_radius,
                hit.ast_orientation,
                hv,
                pivot_vec,
            );

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

    // All despawns done — safe to append crystals; their slots won't be displaced.
    for c in pending_crystals {
        let sprite = spawn_sprite(
            renderer,
            &mut sprite_data.registry,
            build_crystal_sprite_model(),
        );
        commands.push((
            CrystalMarker,
            NewtonBody {
                mass: 0.01,
                pos: c.pos,
                vel: c.vel,
                orientation: DQuat::IDENTITY,
                angular_vel: c.spin,
            },
            sprite,
            Aabb::empty(),
        ));
    }
}

/// Carve a sphere of `radius` voxels centred on `center` in-place. Returns `(is_empty, count)`.
fn carve_sphere(model: &mut SpriteModel, center: IVec3, radius: u32) -> (bool, u32) {
    let r = radius as i32;
    let dims_i = IVec3::from(model.dims.map(|d| d as i32));
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let d = IVec3::new(dx, dy, dz);
                if d.dot(d) > r * r {
                    continue;
                }
                let c = center + d;
                if c.cmpge(IVec3::ZERO).all() && c.cmplt(dims_i).all() {
                    model.set_voxel(c.x as u32, c.y as u32, c.z as u32, None);
                }
            }
        }
    }
    let count = model.colors.len() as u32;
    (count == 0, count)
}

/// Return the subset of `minerals` whose voxel index lies within `radius` of `center`.
fn minerals_in_radius(minerals: &[UVec3], center: IVec3, radius: u32) -> Vec<UVec3> {
    let r2 = (radius as i32).pow(2);
    minerals
        .iter()
        .copied()
        .filter(|&p| (p.as_ivec3() - center).length_squared() <= r2)
        .collect()
}

/// Compute `(delta_vel, delta_omega)` for an asteroid struck by a projectile.
///
/// Uses a solid-sphere moment-of-inertia estimate (I = 2/5·m·r²).
fn hit_impulse(
    proj_vel: DVec3,
    proj_mass: f64,
    ast_mass: f64,
    ast_radius: f64,
    ast_orientation: DQuat,
    hit_voxel: UVec3,
    pivot: DVec3,
) -> (DVec3, DVec3) {
    let hit_local = hit_voxel.as_dvec3() + DVec3::splat(0.5) - pivot;
    let lever = ast_orientation * hit_local;
    let impulse = proj_vel * proj_mass * HIT_IMPULSE_FACTOR;
    let delta_vel = impulse / ast_mass;
    let moment = SOLID_SPHERE_INERTIA * ast_mass * ast_radius * ast_radius;
    let delta_omega = lever.cross(impulse) / moment;
    (delta_vel, delta_omega)
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
) -> Option<UVec3> {
    let local = ast_orientation.inverse() * (proj_pos - ast_pos);
    let vws = model.voxel_world_size as f64;
    let pivot = DVec3::from(model.pivot.map(|p| p as f64));
    let vi = (local / vws + pivot).floor().as_ivec3();
    let dims = IVec3::from(model.dims.map(|d| d as i32));
    if vi.cmplt(IVec3::ZERO).any() || vi.cmpge(dims).any() {
        return None;
    }
    let v = vi.as_uvec3();
    let col = (v.x + v.y * model.dims[0]) as usize;
    let base = col * model.occ_words_per_col as usize;
    let occupied = (model.occupancy[base + v.z as usize / 32] >> (v.z % 32)) & 1 == 1;
    occupied.then_some(v)
}

#[cfg(test)]
mod tests {
    use super::{carve_sphere, hit_impulse, minerals_in_radius, voxel_hit};
    use glam::{DQuat, DVec3, IVec3, UVec3};
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
            Some(UVec3::new(2, 0, 0))
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
            Some(UVec3::new(2, 0, 0))
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

    // ── minerals_in_radius ────────────────────────────────────────────────────

    #[test]
    fn minerals_center_included() {
        let center = IVec3::new(2, 2, 2);
        let pts = vec![UVec3::new(2, 2, 2), UVec3::new(10, 10, 10)];
        assert_eq!(
            minerals_in_radius(&pts, center, 1),
            vec![UVec3::new(2, 2, 2)]
        );
    }

    #[test]
    fn minerals_at_exact_radius_included() {
        // Manhattan distance 2 along one axis, radius 2 → distance² = 4 = r².
        let center = IVec3::new(0, 0, 0);
        let pts = vec![UVec3::new(2, 0, 0)];
        assert_eq!(minerals_in_radius(&pts, center, 2), pts);
    }

    #[test]
    fn minerals_outside_radius_excluded() {
        let center = IVec3::new(0, 0, 0);
        let pts = vec![UVec3::new(3, 0, 0)];
        assert!(minerals_in_radius(&pts, center, 2).is_empty());
    }

    // ── carve_sphere ──────────────────────────────────────────────────────────

    /// Build a solid 5×5×5 model (all voxels occupied).
    fn make_5x5x5() -> SpriteModel {
        let dims = [5u32; 3];
        let n_cols = (dims[0] * dims[1]) as usize; // 25
        let voxels_per_col = dims[2] as usize; // 5
        let mut occupancy = vec![0u32; n_cols];
        for w in &mut occupancy {
            *w = (1u32 << voxels_per_col) - 1; // bits 0–4 set
        }
        let total = n_cols * voxels_per_col; // 125
                                             // Each of the 25 columns holds exactly 5 colors in order.
        let color_offsets: Vec<u32> = (0..=n_cols).map(|i| (i * voxels_per_col) as u32).collect();
        SpriteModel {
            dims,
            occ_words_per_col: 1,
            pivot: [2.5, 2.5, 2.5],
            occupancy,
            colors: vec![0xFF_FF_FF_FF; total],
            dirs: vec![0; total],
            color_offsets,
            voxel_world_size: 1.0,
        }
    }

    #[test]
    fn carve_removes_voxels_in_sphere() {
        let mut model = make_5x5x5();
        let before = model.colors.len();
        // Carve radius 1 at centre (2,2,2) — removes the 7-voxel cross (1+6).
        carve_sphere(&mut model, IVec3::new(2, 2, 2), 1);
        assert!(model.colors.len() < before, "some voxels should be removed");
    }

    #[test]
    fn carve_out_of_bounds_does_not_panic() {
        let mut model = make_5x5x5();
        // Centre at corner; half the sphere is outside the model bounds.
        carve_sphere(&mut model, IVec3::new(0, 0, 0), 4);
    }

    // ── hit_impulse ───────────────────────────────────────────────────────────

    #[test]
    fn hit_impulse_delta_vel_direction() {
        // Projectile moving along +X hits the asteroid centre: lever is zero,
        // so delta_omega should be near zero and delta_vel should be along +X.
        let (dv, domega) = hit_impulse(
            DVec3::X,
            1.0,
            1.0,
            5.0,
            DQuat::IDENTITY,
            UVec3::new(0, 0, 0),
            DVec3::ZERO, // pivot at origin → hit_local = (0.5, 0.5, 0.5)
        );
        assert!(dv.x > 0.0, "delta_vel should point along +X");
        assert!(dv.y.abs() < 1e-10 && dv.z.abs() < 1e-10);
        // Lever is (0.5, 0.5, 0.5), impulse is along X: cross product is non-zero
        // but we just check it's finite.
        assert!(domega.is_finite());
    }

    #[test]
    fn hit_impulse_scales_with_mass_and_speed() {
        // Doubling proj_mass should double delta_vel magnitude.
        let args = |mass: f64| {
            hit_impulse(
                DVec3::X,
                mass,
                10.0,
                5.0,
                DQuat::IDENTITY,
                UVec3::new(5, 5, 5),
                DVec3::splat(5.0),
            )
        };
        let (dv1, _) = args(1.0);
        let (dv2, _) = args(2.0);
        let ratio = dv2.length() / dv1.length();
        assert!((ratio - 2.0).abs() < 1e-10, "ratio was {ratio}");
    }
}
