use legion::{system, systems::CommandBuffer, world::SubWorld, *};
use roxlap_render::SceneRenderer;

use crate::{
    components::{particle::Particle, sprite_id::Sprite},
    systems::sprite::perform_despawn,
    Dt,
};

pub const PARTICLE_LIFETIME: f64 = 10.0;

#[system]
#[write_component(Particle)]
#[read_component(Sprite)]
pub fn particle(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] renderer: &mut SceneRenderer,
    #[resource] dt: &Dt,
) {
    let mut to_despawn: Vec<Entity> = Vec::new();
    {
        let mut q = <(Entity, &mut Particle)>::query();
        for (entity, particle) in q.iter_mut(world) {
            particle.lifetime -= dt.0;
            if particle.lifetime <= 0.0 {
                to_despawn.push(*entity);
            }
        }
    }
    for entity in to_despawn {
        perform_despawn(entity, world, commands, renderer);
    }
}
