pub struct AsteroidMarker;

/// Model-local voxel coordinates of embedded mineral cores. Each point spawns
/// a crystal when the carve sphere reaches it.
pub struct AsteroidMinerals {
    pub points: Vec<glam::UVec3>,
}

/// Voxel count of the asteroid at spawn time; used to trigger auto-destruction
/// when more than 20 % of the volume has been carved away.
pub struct AsteroidVoxelInfo {
    pub initial_count: u32,
}
