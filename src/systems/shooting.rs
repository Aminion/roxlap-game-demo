use bytemuck::Zeroable;
use glam::{DQuat, DVec3};
use legion::{system, systems::CommandBuffer, world::SubWorld, *};
use roxlap_gpu::{GpuRenderer, SpriteInstance, SpriteInstanceTransform};

use crate::{
    components::{
        aabb::Aabb, canon::Canon, miner::Miner, newton_body::NewtonBody, projectile::Projectile,
        sprite_id::SpriteId,
    },
    systems::energy::{Energy, SHOT_COST},
    world::build_projectile_sprite_model,
    SpriteData,
};

const PROJECTILE_SPEED: f64 = 300.0;
const PROJECTILE_LIFETIME: f64 = 6.0;
const CANON_COOLDOWN: f64 = 0.2;
const PROJECTILE_SPAWN_OFFSET: f64 = 3.0;

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
#[write_component(Canon)]
pub fn shooting(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] gpu: &mut GpuRenderer,
    #[resource] sprite_data: &mut SpriteData,
    #[resource] energy: &mut Energy,
) {
    let (spawn_pos, spawn_vel) = {
        let mut miner_q = <(&Miner, &NewtonBody, &mut Canon)>::query();
        let (_, body, canon) = miner_q.iter_mut(world).next().expect("miner missing");
        if !canon.firing || canon.cooldown > 0.0 || energy.current < SHOT_COST {
            return;
        }
        energy.current -= SHOT_COST;
        let forward = (body.orientation * DVec3::NEG_Z).normalize();
        let vel = body.vel + forward * PROJECTILE_SPEED;
        let pos = body.pos + forward * PROJECTILE_SPAWN_OFFSET;
        canon.cooldown = CANON_COOLDOWN;
        (pos, vel)
    };

    let chain_id = sprite_data.registry.add(build_projectile_sprite_model());
    gpu.add_sprite_model(&sprite_data.registry, chain_id);
    let slot = gpu.append_sprite_instances(
        &sprite_data.registry,
        &[SpriteInstance {
            model_id: chain_id,
            transform: SpriteInstanceTransform::zeroed(),
        }],
    );

    commands.push((
        Projectile {
            lifetime: PROJECTILE_LIFETIME,
            chain_id,
        },
        NewtonBody {
            mass: 0.001,
            pos: spawn_pos,
            vel: spawn_vel,
            orientation: DQuat::IDENTITY,
            angular_vel: DVec3::ZERO,
        },
        SpriteId { model_id: slot },
        Aabb { half_extent: 0.5 },
    ));
}
