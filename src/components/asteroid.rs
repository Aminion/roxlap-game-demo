use roxlap_gpu::SpriteModel;

pub struct AsteroidMarker;

/// Permanent record of the GPU model chain id for an asteroid.
/// Present even when the sprite is deactivated (no `SpriteId` component).
pub struct AsteroidChainId(pub u32);

/// CPU-side voxel geometry for an asteroid, kept alive so it can be
/// re-uploaded to the GPU when the asteroid re-enters the presence radius.
pub struct AsteroidModel(pub SpriteModel);
