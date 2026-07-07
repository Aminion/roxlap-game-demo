use glam::{DMat3, DQuat, DVec3};
use legion::World;
use roxlap_core::Camera;
use roxlap_formats::kv6 as kv6_fmt;
use roxlap_gpu::{build_sprite_model, SpriteModelRegistry};
use roxlap_render::{Material, Rgb, SceneRenderer, SpriteModelId};

use crate::components::{
    aabb::Aabb, camera::CameraComponent, cannon::Cannon, miner::Miner, newton_body::NewtonBody,
    presence_position::PresencePosition, sprite_id::Sprite, thruster::ThrusterBank,
};
use crate::sprites::{build_crystal, build_particle, build_projectile};
use crate::systems::sprite::sprite_model_to_kv6;

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
}

pub struct MinerModel {
    pub model_id: SpriteModelId,
    pub chain_id: u32,
    /// Half the largest KV6 dimension — used as the ship's point-light radius.
    pub radius: f32,
    /// Half the KV6 z-depth — distance from body center to the nose along the forward axis.
    pub nose_offset: f64,
}

pub fn register_miner_model(
    renderer: &mut SceneRenderer,
    registry: &mut SpriteModelRegistry,
) -> MinerModel {
    static KV6_BYTES: &[u8] = include_bytes!("../model.kv6");
    let mut kv6 = kv6_fmt::parse(KV6_BYTES).expect("model.kv6 parse failed");
    // Override whatever pivot the file stores with the geometric centre so
    // body.pos always maps to the model's centre regardless of its dimensions.
    kv6.xpiv = kv6.xsiz as f32 * 0.5;
    kv6.ypiv = kv6.ysiz as f32 * 0.5;
    kv6.zpiv = kv6.zsiz as f32 * 0.5;
    let radius = kv6.xsiz.max(kv6.ysiz).max(kv6.zsiz) as f32 * 0.5;
    let nose_offset = kv6.zsiz as f64 * 0.5;
    let model_id = renderer.add_sprite_model(&kv6);
    let chain_id = registry.add(build_sprite_model(&kv6));
    MinerModel {
        model_id,
        chain_id,
        radius,
        nose_offset,
    }
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
        renderer.add_sprite_model_with_materials(&crystal_kv6, &[(Rgb(0xFF_30_30), 1)]);

    // Particles are driven directly by the renderer's ParticleSystem from the
    // model id alone, so the CPU registry never needs a particle chain entry.
    let particle_kv6 = sprite_model_to_kv6(&build_particle());
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
        },
    )
}

pub fn populate_world(world: &mut World, renderer: &mut SceneRenderer, miner_model: &MinerModel) {
    spawn_miner(world, renderer, miner_model);
}

const MINER_PITCH: f64 = 0.8;
const MINER_SPAWN_OFFSET_X: f64 = 70.0;
const MINER_SPAWN_HEIGHT: f64 = 100.0;

/// Spawn attitude: right = world +Y, nose pitched MINER_PITCH below the
/// horizon toward +X (world is z-down, so forward gains a +z component).
fn miner_orientation() -> DQuat {
    let (sp, cp) = (MINER_PITCH.sin(), MINER_PITCH.cos());
    DQuat::from_mat3(&DMat3::from_cols(
        DVec3::Y,
        DVec3::new(sp, 0.0, -cp),
        DVec3::new(-cp, 0.0, -sp),
    ))
}

pub fn miner_initial_forward() -> DVec3 {
    miner_orientation() * DVec3::NEG_Z
}

fn spawn_miner(world: &mut World, renderer: &mut SceneRenderer, miner_model: &MinerModel) {
    let orientation = miner_orientation();
    let pos = DVec3::new(-MINER_SPAWN_OFFSET_X, 0.0, -MINER_SPAWN_HEIGHT);
    let instance_id = renderer
        .add_sprite_instance_posed(
            miner_model.model_id,
            roxlap_render::DynSpriteTransform::default(),
        )
        .expect("miner sprite model is live");
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
        Cannon { cooldown: 0.0 },
        Sprite {
            chain_id: miner_model.chain_id,
            model_id: miner_model.model_id,
            instance_id,
            owns_model: false,
        },
        Aabb::empty(),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miner_orientation_is_unit_quaternion() {
        // from_mat3 on a non-rotation (det ≠ +1) basis silently yields a
        // non-unit quaternion; this catches a left-handed column set.
        let q = miner_orientation();
        assert!(
            (q.length() - 1.0).abs() < 1e-12,
            "|q| = {} — basis matrix is not a proper rotation",
            q.length()
        );
    }

    #[test]
    fn miner_orientation_matches_basis_columns() {
        let (sp, cp) = (MINER_PITCH.sin(), MINER_PITCH.cos());
        let q = miner_orientation();
        let right = q * DVec3::X;
        let up = q * DVec3::Y;
        let fwd = q * DVec3::NEG_Z;
        assert!((right - DVec3::Y).length() < 1e-12, "right: {right:?}");
        assert!(
            (up - DVec3::new(sp, 0.0, -cp)).length() < 1e-12,
            "up: {up:?}"
        );
        assert!(
            (fwd - DVec3::new(cp, 0.0, sp)).length() < 1e-12,
            "forward: {fwd:?}"
        );
    }
}
