use glam::DVec3;
use legion::{system, world::SubWorld, *};

use crate::{
    components::{aabb::Aabb, newton_body::NewtonBody, sprite_id::Sprite},
    SpriteData,
};

#[system]
#[read_component(Sprite)]
#[read_component(NewtonBody)]
#[write_component(Aabb)]
pub fn aabb_update(world: &mut SubWorld, #[resource] sprite_data: &SpriteData) {
    let mut q = <(&Sprite, &NewtonBody, &mut Aabb)>::query();
    for (sprite, body, aabb) in q.iter_mut(world) {
        let model = sprite_data.registry.model(sprite.chain_id);
        let vws = model.voxel_world_size as f64;
        let pivot = DVec3::from(model.pivot.map(|p| p as f64));
        let dims = DVec3::new(
            model.dims[0] as f64,
            model.dims[1] as f64,
            model.dims[2] as f64,
        );
        // Model box in body-local space: runs from -pivot*vws to (dims-pivot)*vws.
        let local_min = -pivot * vws;
        let local_max = (dims - pivot) * vws;

        // OBB→AABB: transform all 8 corners through orientation, take per-axis min/max.
        let mut world_min = DVec3::splat(f64::INFINITY);
        let mut world_max = DVec3::splat(f64::NEG_INFINITY);
        for sx in [local_min.x, local_max.x] {
            for sy in [local_min.y, local_max.y] {
                for sz in [local_min.z, local_max.z] {
                    let corner = body.pos + body.orientation * DVec3::new(sx, sy, sz);
                    world_min = world_min.min(corner);
                    world_max = world_max.max(corner);
                }
            }
        }
        aabb.min = world_min;
        aabb.max = world_max;
    }
}
