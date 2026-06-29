use glam::Vec3;
use legion::{system, systems::CommandBuffer, world::SubWorld, *};
use roxlap_render::SceneRenderer;

use crate::{
    components::particle::{Particle, ParticleGroup},
    systems::sprite::perform_despawn,
    Dt,
};

/// Exponential scale decay rate per second: scale *= exp(-DECAY_RATE·dt) each frame.
pub const PARTICLE_DECAY_RATE: f32 = 1.0;
/// Side length of the shared particle cube model in voxels; sets initial world-space scale.
pub const PARTICLE_MODEL_DIM: f32 = 3.0;
const DESPAWN_THRESHOLD: f32 = 0.005;

#[system]
#[write_component(ParticleGroup)]
#[write_component(Particle)]
pub fn particle(
    world: &mut SubWorld,
    commands: &mut CommandBuffer,
    #[resource] renderer: &mut SceneRenderer,
    #[resource] dt: &Dt,
) {
    let decay_factor = (-PARTICLE_DECAY_RATE * dt.0 as f32).exp();
    let mut scale_updates: Vec<(Entity, Vec3)> = Vec::new();
    let mut to_despawn: Vec<(Entity, Vec<Entity>)> = Vec::new();
    {
        let mut q = <(Entity, &mut ParticleGroup)>::query();
        for (entity, group) in q.iter_mut(world) {
            group.scale *= decay_factor;
            if group.scale.max_element() < DESPAWN_THRESHOLD {
                to_despawn.push((*entity, std::mem::take(&mut group.members)));
            } else {
                for &member in &group.members {
                    scale_updates.push((member, group.scale));
                }
            }
        }
    }
    for (member, scale) in scale_updates {
        if let Ok(mut entry) = world.entry_mut(member) {
            if let Ok(p) = entry.get_component_mut::<Particle>() {
                p.scale = scale;
            }
        }
    }
    for (group_entity, members) in to_despawn {
        for member in members {
            perform_despawn(member, world, commands, renderer);
        }
        commands.remove(group_entity);
    }
}
