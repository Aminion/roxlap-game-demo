use roxlap_render::{SpriteInstanceId, SpriteModelId};

/// Renderer + CPU-registry binding for a sprite entity.
///
/// `chain_id` is the dense occupancy registry index (stable; used by game
/// logic for hit detection and carving).  `model_id` / `instance_id` are
/// the stable `SceneRenderer` handles — swap-remove is transparent to callers.
pub struct Sprite {
    pub chain_id: u32,
    pub model_id: SpriteModelId,
    pub instance_id: SpriteInstanceId,
}
