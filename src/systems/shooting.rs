use glam::{DQuat, DVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, *};
use roxlap_render::{BillboardLighting, SceneRenderer};

use crate::{
    components::{cannon::Cannon, miner::Miner, newton_body::NewtonBody, projectile::Projectile},
    systems::energy::{Energy, SHOT_COST},
    world::{spawn_shared_instance, MinerModel, ProjectileModel},
    Dt,
};

const PROJECTILE_SPEED: f64 = 300.0;
const PROJECTILE_MASS: f64 = 0.001;
const PROJECTILE_LIFETIME: f64 = 6.0;
const CANNON_COOLDOWN: f64 = 0.2;

#[system]
#[read_component(Miner)]
#[write_component(NewtonBody)]
#[write_component(Cannon)]
pub fn shooting(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] renderer: &mut SceneRenderer,
    #[resource] proj_model: &ProjectileModel,
    #[resource] miner_model: &MinerModel,
    #[resource] energy: &mut Energy,
    #[resource] dt: &Dt,
) {
    // Phase 1: check firing conditions and capture miner state.
    let (miner_pos, miner_vel, miner_mass, shoot_dir) = {
        let mut q = <(&Miner, &mut NewtonBody, &mut Cannon)>::query();
        let (_, body, cannon) = q.iter_mut(world).next().expect("miner missing");
        cannon.cooldown = (cannon.cooldown - dt.0).max(0.0);
        if !cannon.firing || cannon.cooldown > 0.0 || energy.current < SHOT_COST {
            return;
        }
        energy.current -= SHOT_COST;
        cannon.cooldown = CANNON_COOLDOWN;
        let fwd = body.orientation * DVec3::NEG_Z;
        (body.pos, body.vel, body.mass, fwd)
    };

    // Phase 2: apply recoil, spawn projectile from the nose.
    let spawn_pos = miner_pos + shoot_dir * miner_model.nose_offset;
    let spawn_vel = miner_vel + shoot_dir * PROJECTILE_SPEED;

    {
        let mut q = <(&Miner, &mut NewtonBody)>::query();
        let (_, body) = q.iter_mut(world).next().expect("miner missing");
        body.vel -= shoot_dir * (PROJECTILE_MASS * PROJECTILE_SPEED / miner_mass);
    }

    let sprite = spawn_shared_instance(renderer, proj_model.model_id, proj_model.chain_id);
    // Tracers glow: render the projectile at full intensity, ignoring the
    // lighting rig (otherwise the shot darkens when facing away from the sun).
    renderer.set_sprite_instance_lighting(sprite.instance_id, BillboardLighting::FullBright);
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
