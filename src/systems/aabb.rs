use glam::{DMat3, DQuat, DVec3};
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
        let local_min = -pivot * vws;
        let local_max = (dims - pivot) * vws;
        *aabb = obb_to_aabb(local_min, local_max, body.pos, body.orientation);
    }
}

/// Converts an oriented bounding box to a world-space AABB.
///
/// Projects the local half-extents through |R| (absolute rotation matrix) to
/// get tight world-space half-extents — the standard OBB→AABB technique.
fn obb_to_aabb(local_min: DVec3, local_max: DVec3, pos: DVec3, orientation: DQuat) -> Aabb {
    let mat = DMat3::from_quat(orientation);
    let half = (local_max - local_min) * 0.5;
    let center = pos + orientation * ((local_min + local_max) * 0.5);
    // world_half_i = Σ_j |R_ij|·half_j — rows of |R|, not columns: columns
    // would compute |Rᵀ|·half, which underestimates for asymmetric rotations.
    let world_half = mat.abs() * half;
    Aabb {
        min: center - world_half,
        max: center + world_half,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn obb_to_aabb_identity_orientation() {
        // Identity rotation: world AABB equals the local box translated to pos.
        let aabb = obb_to_aabb(
            DVec3::new(-1.0, -2.0, -3.0),
            DVec3::new(1.0, 2.0, 3.0),
            DVec3::ZERO,
            DQuat::IDENTITY,
        );
        assert!((aabb.min - DVec3::new(-1.0, -2.0, -3.0)).length() < 1e-10);
        assert!((aabb.max - DVec3::new(1.0, 2.0, 3.0)).length() < 1e-10);
    }

    #[test]
    fn obb_to_aabb_translation() {
        let pos = DVec3::new(10.0, 20.0, 30.0);
        let aabb = obb_to_aabb(DVec3::splat(-1.0), DVec3::splat(1.0), pos, DQuat::IDENTITY);
        assert!((aabb.min - (pos - DVec3::splat(1.0))).length() < 1e-10);
        assert!((aabb.max - (pos + DVec3::splat(1.0))).length() < 1e-10);
    }

    #[test]
    fn obb_to_aabb_rot90_y_swaps_x_and_z() {
        // Box [-1,-1,-2]..[1,1,2], half = (1,1,2).
        // 90° Y rotation maps X→-Z, Z→+X, so world half-extents become (2,1,1).
        let aabb = obb_to_aabb(
            DVec3::new(-1.0, -1.0, -2.0),
            DVec3::new(1.0, 1.0, 2.0),
            DVec3::ZERO,
            DQuat::from_rotation_y(FRAC_PI_2),
        );
        assert!(
            (aabb.min - DVec3::new(-2.0, -1.0, -1.0)).length() < 1e-10,
            "min: {:?}",
            aabb.min
        );
        assert!(
            (aabb.max - DVec3::new(2.0, 1.0, 1.0)).length() < 1e-10,
            "max: {:?}",
            aabb.max
        );
    }

    #[test]
    fn obb_to_aabb_matches_corner_hull_for_asymmetric_rotation() {
        // Compound rotation whose |R| is not symmetric — distinguishes rows
        // from columns. Ground truth: AABB of the 8 transformed box corners.
        let orientation = DQuat::from_rotation_x(1.2) * DQuat::from_rotation_z(0.7);
        let local_min = DVec3::new(-1.0, -2.0, -3.0);
        let local_max = DVec3::new(1.0, 2.0, 3.0);
        let pos = DVec3::new(4.0, -5.0, 6.0);

        let mut hull_min = DVec3::splat(f64::INFINITY);
        let mut hull_max = DVec3::splat(f64::NEG_INFINITY);
        for i in 0..8 {
            let corner = DVec3::new(
                if i & 1 == 0 { local_min.x } else { local_max.x },
                if i & 2 == 0 { local_min.y } else { local_max.y },
                if i & 4 == 0 { local_min.z } else { local_max.z },
            );
            let world = pos + orientation * corner;
            hull_min = hull_min.min(world);
            hull_max = hull_max.max(world);
        }

        let aabb = obb_to_aabb(local_min, local_max, pos, orientation);
        assert!(
            (aabb.min - hull_min).length() < 1e-10,
            "min: {:?} vs hull {:?}",
            aabb.min,
            hull_min
        );
        assert!(
            (aabb.max - hull_max).length() < 1e-10,
            "max: {:?} vs hull {:?}",
            aabb.max,
            hull_max
        );
    }

    #[test]
    fn obb_to_aabb_is_symmetric() {
        // Result must satisfy min <= max on all axes.
        let aabb = obb_to_aabb(
            DVec3::splat(-3.0),
            DVec3::splat(3.0),
            DVec3::new(1.0, 2.0, 3.0),
            DQuat::from_rotation_x(1.2) * DQuat::from_rotation_z(0.7),
        );
        assert!(aabb.min.x <= aabb.max.x);
        assert!(aabb.min.y <= aabb.max.y);
        assert!(aabb.min.z <= aabb.max.z);
    }
}
