use legion::{systems::CommandBuffer, world::SubWorld, Entity, EntityStore};
use roxlap_render::SceneRenderer;

use crate::components::sprite_id::Sprite;

/// Remove a single entity from the renderer, ECS, and all bookkeeping.
///
/// Does NOT touch `VisitedChunks` — callers that want the chunk to be
/// re-populatable (distance-based unload) must remove it from `visited`
/// themselves.
pub fn perform_despawn(
    entity: Entity,
    world: &SubWorld,
    commands: &mut CommandBuffer,
    renderer: &mut SceneRenderer,
) {
    let Ok(entry) = world.entry_ref(entity) else {
        return;
    };
    let Ok(sprite) = entry.get_component::<Sprite>() else {
        return;
    };
    renderer.remove_sprite_instance(sprite.instance_id);
    if sprite.owns_model {
        renderer.remove_sprite_model(sprite.model_id);
    }
    commands.remove(entity);
}
