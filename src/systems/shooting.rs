use glam::{DQuat, DVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, *};
use roxlap_render::SceneRenderer;

use crate::{
    components::{cannon::Cannon, miner::Miner, newton_body::NewtonBody, projectile::Projectile},
    systems::energy::{Energy, SHOT_COST},
    world::{spawn_shared_instance, ProjectileModel},
    Dt,
};

const PROJECTILE_SPEED: f64 = 300.0;
const PROJECTILE_LIFETIME: f64 = 6.0;
const CANNON_COOLDOWN: f64 = 0.2;
const PROJECTILE_SPAWN_OFFSET: f64 = 3.0;

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[write_component(Cannon)]
pub fn shooting(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] renderer: &mut SceneRenderer,
    #[resource] proj_model: &ProjectileModel,
    #[resource] energy: &mut Energy,
    #[resource] dt: &Dt,
) {
    let (spawn_pos, spawn_vel) = {
        let mut miner_q = <(&Miner, &NewtonBody, &mut Cannon)>::query();
        let (_, body, canon) = miner_q.iter_mut(world).next().expect("miner missing");
        canon.cooldown = (canon.cooldown - dt.0).max(0.0);
        if !canon.firing || canon.cooldown > 0.0 || energy.current < SHOT_COST {
            return;
        }
        energy.current -= SHOT_COST;
        let forward = (body.orientation * DVec3::NEG_Z).normalize();
        let vel = body.vel + forward * PROJECTILE_SPEED;
        let pos = body.pos + forward * PROJECTILE_SPAWN_OFFSET;
        canon.cooldown = CANNON_COOLDOWN;
        (pos, vel)
    };

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
