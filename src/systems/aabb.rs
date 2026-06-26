use glam::{DMat3, DVec3};
use legion::{system, world::SubWorld, *};

use roxlap_gpu::SpriteModelRegistry;

use crate::components::{aabb::Aabb, newton_body::NewtonBody, sprite_id::Sprite};

#[system]
#[read_component(Sprite)]
#[read_component(NewtonBody)]
#[write_component(Aabb)]
pub fn aabb_update(world: &mut SubWorld, #[resource] registry: &SpriteModelRegistry) {
    let mut q = <(&Sprite, &NewtonBody, &mut Aabb)>::query();
    for (sprite, body, aabb) in q.iter_mut(world) {
        let model = registry.model(sprite.chain_id);
        let vws = model.voxel_world_size as f64;
        let pivot = DVec3::from(model.pivot.map(|p| p as f64));
        let dims = DVec3::from(model.dims.map(|d| d as f64));
        // Model box in body-local space: runs from -pivot*vws to (dims-pivot)*vws.
        let local_min = -pivot * vws;
        let local_max = (dims - pivot) * vws;

        // OBB→AABB: project half-extents through |R| to get world half-extents.
        let mat = DMat3::from_quat(body.orientation);
        let half = (local_max - local_min) * 0.5;
        let center = body.pos + body.orientation * ((local_min + local_max) * 0.5);
        let world_half = DVec3::new(
            mat.col(0).abs().dot(half),
            mat.col(1).abs().dot(half),
            mat.col(2).abs().dot(half),
        );
        aabb.min = center - world_half;
        aabb.max = center + world_half;
    }
}
