use roxlap_render::{SpriteInstanceId, SpriteModelId};

/// Renderer + CPU-registry binding for a sprite entity.
///
/// `chain_id` is the dense occupancy registry index (stable; used by game
/// logic for hit detection and carving).  `model_id` / `instance_id` are
/// the stable `SceneRenderer` handles — swap-remove is transparent to callers.
/// `owns_model` is false for entities that share a pre-registered model (projectiles,
/// crystals); `perform_despawn` skips `remove_sprite_model` for those.
pub struct Sprite {
    pub chain_id: u32,
    pub model_id: SpriteModelId,
    pub instance_id: SpriteInstanceId,
    pub owns_model: bool,
}
