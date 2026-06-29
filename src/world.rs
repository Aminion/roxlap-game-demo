use std::f32::consts::{PI, TAU};

use glam::{DMat3, DQuat, DVec3, IVec2, Vec2};
use legion::World;
use rand::{
    distr::weighted::WeightedIndex, distr::Distribution, rngs::StdRng, RngExt, SeedableRng,
};
use roxlap_core::Camera;
use roxlap_gpu::{SpriteModel, SpriteModelRegistry};

use crate::sprites::{build_crystal, build_particle, build_projectile};
use roxlap_render::{DynSpriteTransform, Kv6, Material, SceneRenderer, SpriteModelId};

use crate::components::{
    camera::CameraComponent, cannon::Cannon, miner::Miner, newton_body::NewtonBody,
    presence_position::PresencePosition, sprite_id::Sprite, thruster::ThrusterBank,
};

/// Generate a 1024×512 equirectangular star panorama as RGBA bytes.
///
/// Stars are placed with uniform solid-angle distribution so the sky
/// looks even from any direction (avoids pole crowding that a naive
/// random-pixel scatter would produce).
pub fn generate_star_sky(seed: u64) -> (Vec<u8>, u32, u32) {
    // u=elevation covers [0,π], v=azimuth covers [0,2π]; set H=2W so
    // both axes have equal angular resolution and stars appear round.
    const W: u32 = 512;
    const H: u32 = 1024;
    const STAR_COUNT: u32 = 1024;

    let mut pixels = vec![0u8; (W * H * 4) as usize];
    let mut rng = StdRng::seed_from_u64(seed);
    let color_dist = WeightedIndex::new([2, 2, 6]).unwrap(); // 20% red, 20% blue, 60% white
    let size_dist = WeightedIndex::new([6, 3, 1]).unwrap(); // 60% sz2, 30% sz3, 10% sz4

    for _ in 0..STAR_COUNT {
        // Uniform solid-angle distribution: cos(polar) uniform in [-1, 1].
        let cos_theta: f32 = rng.random_range(-1.0f32..=1.0);
        let phi: f32 = rng.random_range(0.0f32..TAU);

        // Match the UV convention in scene_dda.wgsl:
        //   u = elevation = acos(-dir.z) / π  → texture column (x)
        //   v = azimuth  = atan2(x,y)/(2π)+.5 → texture row    (y)
        let uv = Vec2::new((-cos_theta).acos() / PI, phi / TAU);
        let center = IVec2::new(
            (uv.x * W as f32).min(W as f32 - 1.0) as i32,
            (uv.y * H as f32) as i32 % H as i32,
        );

        let brightness: u8 = rng.random_range(160u8..=220);
        let b = brightness as f32;

        // Color: 20% red-biased, 20% blue-biased, 60% white.
        let (r, g, bl) = match color_dist.sample(&mut rng) {
            0 => ((b * 1.0) as u8, (b * 0.80) as u8, (b * 0.75) as u8),
            1 => ((b * 0.75) as u8, (b * 0.85) as u8, (b * 1.0) as u8),
            _ => (brightness, brightness, brightness),
        };

        // Size: 2, 3, or 4 pixels — larger than before to survive bilinear blur.
        let size: i32 = match size_dist.sample(&mut rng) {
            0 => 2,
            1 => 3,
            _ => 4,
        };

        // In equirectangular, azimuth pixels compress by sin(elevation) near the poles.
        // Stretch the star in the azimuth (row) direction by 1/sin(elevation) so it
        // appears round on screen regardless of where on the sphere it sits.
        let sin_elev = (uv.x * PI).sin();
        let width_v = ((size as f32 / sin_elev).round() as i32).max(size);
        let half = IVec2::new(size / 2, width_v / 2);

        for dx in 0..size {
            // elevation (column) direction
            for dy in 0..width_v {
                // azimuth (row) direction — stretched
                let pixel = IVec2::new(
                    (center.x + dx - half.x).clamp(0, W as i32 - 1),
                    (center.y + dy - half.y).rem_euclid(H as i32),
                );
                let i = ((pixel.y as u32 * W + pixel.x as u32) * 4) as usize;
                pixels[i] = r;
                pixels[i + 1] = g;
                pixels[i + 2] = bl;
                pixels[i + 3] = 255;
            }
        }
    }

    (pixels, W, H)
}

/// Convert a dense-occupancy `SpriteModel` into a surface-only `Kv6` for the renderer.
pub fn sprite_model_to_kv6(model: &SpriteModel) -> Kv6 {
    let [mx, my, mz] = model.dims;
    let occ = model.occ_words_per_col as usize;
    Kv6::from_fn_shaded(mx, my, mz, |x, y, z| {
        let col = (x + y * mx) as usize;
        let base = col * occ;
        let z_word = z as usize / 32;
        let z_bit = z % 32;
        if (model.occupancy[base + z_word] >> z_bit) & 1 == 0 {
            return None;
        }
        let col_start = model.color_offsets[col] as usize;
        let mut rank = 0usize;
        for w in 0..z_word {
            rank += model.occupancy[base + w].count_ones() as usize;
        }
        let below_mask = (1u32 << z_bit) - 1;
        rank += (model.occupancy[base + z_word] & below_mask).count_ones() as usize;
        Some(model.colors[col_start + rank])
    })
}

pub struct ProjectileModel {
    pub model_id: SpriteModelId,
    pub chain_id: u32,
}

pub struct CrystalModel {
    pub model_id: SpriteModelId,
    pub chain_id: u32,
}

pub struct ParticleModel {
    pub model_id: SpriteModelId,
    pub chain_id: u32,
}

/// Register shared models for projectiles, crystals, and debris particles.
/// Called once at startup and again after every `restart_world`.
pub fn register_shared_sprites(
    renderer: &mut SceneRenderer,
    registry: &mut SpriteModelRegistry,
) -> (ProjectileModel, CrystalModel, ParticleModel) {
    let proj_chain_id = registry.add(build_projectile());
    let proj_kv6 = sprite_model_to_kv6(registry.model(proj_chain_id));
    let proj_model_id = renderer.add_sprite_model(&proj_kv6);

    renderer.define_material(1, Material::alpha_blend(160));
    let crystal_chain_id = registry.add(build_crystal());
    let crystal_kv6 = sprite_model_to_kv6(registry.model(crystal_chain_id));
    let crystal_model_id =
        renderer.add_sprite_model_with_materials(&crystal_kv6, &[(0xFF_30_30, 1)]);

    let particle_chain_id = registry.add(build_particle());
    let particle_kv6 = sprite_model_to_kv6(registry.model(particle_chain_id));
    let particle_model_id = renderer.add_sprite_model(&particle_kv6);

    (
        ProjectileModel {
            model_id: proj_model_id,
            chain_id: proj_chain_id,
        },
        CrystalModel {
            model_id: crystal_model_id,
            chain_id: crystal_chain_id,
        },
        ParticleModel {
            model_id: particle_model_id,
            chain_id: particle_chain_id,
        },
    )
}

/// Spawn an additional instance of a pre-registered shared model (no model ownership).
pub fn spawn_shared_instance(
    renderer: &mut SceneRenderer,
    model_id: SpriteModelId,
    chain_id: u32,
) -> Sprite {
    let instance_id = renderer.add_sprite_instance_posed(model_id, DynSpriteTransform::default());
    Sprite {
        chain_id,
        model_id,
        instance_id,
        owns_model: false,
    }
}

/// Register a sprite model with both the CPU registry and the renderer.
pub fn spawn_sprite(
    renderer: &mut SceneRenderer,
    registry: &mut SpriteModelRegistry,
    model: SpriteModel,
) -> Sprite {
    let chain_id = registry.add(model);
    let kv6 = sprite_model_to_kv6(registry.model(chain_id));
    let model_id = renderer.add_sprite_model(&kv6);
    let instance_id = renderer.add_sprite_instance_posed(model_id, DynSpriteTransform::default());
    Sprite {
        chain_id,
        model_id,
        instance_id,
        owns_model: true,
    }
}

pub fn populate_world(world: &mut World) {
    spawn_miner(world);
}

const MINER_PITCH: f64 = 0.8;
const MINER_SPAWN_OFFSET_X: f64 = 70.0;
const MINER_SPAWN_HEIGHT: f64 = 100.0;

fn miner_orientation() -> DQuat {
    let (sp, cp) = (MINER_PITCH.sin(), MINER_PITCH.cos());
    DQuat::from_mat3(&DMat3::from_cols(
        DVec3::Y,
        DVec3::new(-sp, 0.0, cp),
        DVec3::new(-cp, 0.0, -sp),
    ))
    .normalize()
}

pub fn miner_initial_forward() -> DVec3 {
    miner_orientation() * DVec3::NEG_Z
}

fn spawn_miner(world: &mut World) {
    let orientation = miner_orientation();
    let pos = DVec3::new(-MINER_SPAWN_OFFSET_X, 0.0, -MINER_SPAWN_HEIGHT);
    // CameraComponent is overwritten by camera_update_system before the first render,
    // so the initial values are placeholders.
    world.push((
        Miner,
        NewtonBody {
            mass: 1.0,
            pos,
            vel: DVec3::ZERO,
            orientation,
            angular_vel: DVec3::ZERO,
        },
        CameraComponent(Camera {
            pos: [0.0; 3],
            forward: [0.0, 0.0, -1.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
        }),
        // mass=1.0 kg, radius=1.0 m, rot=0.6 N → 3.0 rad/s² max; lin=5.0 N → 5.0 m/s² max
        ThrusterBank::new(1.0, 1.0, 0.6, 5.0),
        // Infinity forces the distance check to fire on frame 1, generating the initial chunks.
        PresencePosition(DVec3::splat(f64::INFINITY)),
        Cannon {
            firing: false,
            cooldown: 0.0,
        },
    ));
}
