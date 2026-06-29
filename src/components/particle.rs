use glam::Vec3;
use legion::Entity;

pub struct Particle {
    pub scale: Vec3,
}

/// Owns the shared decaying scale for all debris from one projectile hit.
/// All `members` are batch-despawned when `scale` drops below the threshold.
pub struct ParticleGroup {
    pub scale: Vec3,
    pub members: Vec<Entity>,
}
