use glam::DVec3;

/// World-space point used to determine which asteroid chunks to generate, load, or unload.
pub struct PresencePosition(pub DVec3);
