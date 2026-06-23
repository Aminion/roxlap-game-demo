/// GPU sprite binding: instance buffer slot and model chain ID.
///
/// `slot` is the index into the GPU instance buffer (updated on swap-remove).
/// `chain_id` is the model registry index (stable until `compact_sprite_models`).
pub struct Sprite {
    pub slot: u32,
    pub chain_id: u32,
}
