/// Axis-aligned bounding box component, stored as a uniform half-extent in world units.
/// For spherical asteroid models the AABB doesn't change with orientation.
pub struct Aabb {
    pub half_extent: f32,
}
