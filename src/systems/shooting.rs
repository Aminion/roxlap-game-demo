use glam::{DQuat, DVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, *};
use roxlap_render::SceneRenderer;

use crate::{
    components::{
        aabb::Aabb, camera::CameraComponent, cannon::Cannon, miner::Miner, newton_body::NewtonBody,
        projectile::Projectile,
    },
    math::ray_aabb,
    systems::energy::{Energy, SHOT_COST},
    world::{spawn_shared_instance, ProjectileModel},
    Dt, LoadedAsteroids,
};

const PROJECTILE_SPEED: f64 = 300.0;
const PROJECTILE_MASS: f64 = 0.001;
const PROJECTILE_LIFETIME: f64 = 6.0;
const CANNON_COOLDOWN: f64 = 0.2;
const AIM_FALLBACK_DIST: f64 = 500.0;

#[system]
#[read_component(Miner)]
#[read_component(CameraComponent)]
#[write_component(NewtonBody)]
#[read_component(Aabb)]
#[write_component(Cannon)]
pub fn shooting(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] renderer: &mut SceneRenderer,
    #[resource] proj_model: &ProjectileModel,
    #[resource] energy: &mut Energy,
    #[resource] loaded: &LoadedAsteroids,
    #[resource] dt: &Dt,
) {
    // Phase 1: check firing conditions and capture miner state.
    let (miner_pos, miner_vel, miner_mass, cam_pos, forward) = {
        let mut q = <(&Miner, &mut NewtonBody, &mut Cannon, &CameraComponent)>::query();
        let (_, body, cannon, cam) = q.iter_mut(world).next().expect("miner missing");
        cannon.cooldown = (cannon.cooldown - dt.0).max(0.0);
        if !cannon.firing || cannon.cooldown > 0.0 || energy.current < SHOT_COST {
            return;
        }
        energy.current -= SHOT_COST;
        cannon.cooldown = CANNON_COOLDOWN;
        let fwd = (body.orientation * DVec3::NEG_Z).normalize();
        (body.pos, body.vel, body.mass, DVec3::from(cam.0.pos), fwd)
    };

    // Phase 2: cast camera-center ray against asteroid AABBs to find aim point.
    let aim_point = {
        let mut best_t = f64::INFINITY;
        for &entity in &loaded.0 {
            if let Ok(entry) = world.entry_ref(entity) {
                if let Ok(aabb) = entry.get_component::<Aabb>() {
                    if let Some(t) = ray_aabb(cam_pos, forward, aabb.min, aabb.max) {
                        let hit = cam_pos + forward * t;
                        // Discard hits behind the miner (camera inside an asteroid AABB).
                        if (hit - miner_pos).dot(forward) > 0.0 && t < best_t {
                            best_t = t;
                        }
                    }
                }
            }
        }
        if best_t.is_finite() {
            cam_pos + forward * best_t
        } else {
            cam_pos + forward * AIM_FALLBACK_DIST
        }
    };

    // Phase 3: compute shoot direction, apply recoil, spawn projectile.
    let spawn_pos = miner_pos;
    let shoot_dir = (aim_point - spawn_pos).try_normalize().unwrap_or(forward);
    let spawn_vel = miner_vel + shoot_dir * PROJECTILE_SPEED;

    // Recoil: conservation of momentum — kick miner opposite to shot direction.
    {
        let mut q = <(&Miner, &mut NewtonBody)>::query();
        let (_, body) = q.iter_mut(world).next().expect("miner missing");
        body.vel -= shoot_dir * (PROJECTILE_MASS * PROJECTILE_SPEED / miner_mass);
    }

    let sprite = spawn_shared_instance(renderer, proj_model.model_id, proj_model.chain_id);
    commands.push((
        Projectile {
            lifetime: PROJECTILE_LIFETIME,
        },
        NewtonBody {
            mass: 0.001,
            pos: spawn_pos,
            vel: spawn_vel,
            orientation: DQuat::IDENTITY,
            angular_vel: DVec3::ZERO,
        },
        sprite,
    ));
}
