use glam::DVec3;

/// Remove the component of `v` along `axis` (assumed unit vector).
#[inline]
pub fn reject(v: DVec3, axis: DVec3) -> DVec3 {
    v - axis * v.dot(axis)
}

/// Bijective u64→u64 hash (xor-shift-multiply finalizer from SplitMix64).
/// Strong avalanche: a 1-bit input change flips ~32 output bits, so
/// splitmix64(h+0)..splitmix64(h+N) are statistically independent — safe
/// to use as separate random streams without a sequential PRNG.
pub fn splitmix64(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    h
}

/// Maps the top 24 bits of a hash word to [-1, 1).
pub fn hash_to_signed(v: u64) -> f64 {
    (v >> 40) as f64 / 8_388_608.0 - 1.0
}

/// Slab-method ray–AABB test. Returns the entry t along `ray_dir`, or `None`.
pub fn ray_aabb(ray_origin: DVec3, ray_dir: DVec3, min: DVec3, max: DVec3) -> Option<f64> {
    let inv = ray_dir.recip();
    let t1 = (min - ray_origin) * inv;
    let t2 = (max - ray_origin) * inv;
    let t_min = t1.min(t2).max_element();
    let t_max = t1.max(t2).min_element();
    if t_max < t_min || t_max < 0.0 {
        return None;
    }
    Some(if t_min >= 0.0 { t_min } else { t_max })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── splitmix64 ────────────────────────────────────────────────────────────

    #[test]
    fn splitmix64_deterministic() {
        assert_eq!(splitmix64(0), splitmix64(0));
        assert_eq!(splitmix64(42), splitmix64(42));
    }

    #[test]
    fn splitmix64_adjacent_seeds_differ() {
        assert_ne!(splitmix64(0), splitmix64(1));
        assert_ne!(splitmix64(100), splitmix64(101));
    }

    // ── hash_to_signed ────────────────────────────────────────────────────────

    #[test]
    fn hash_to_signed_range() {
        for seed in [0u64, 1, 42, u64::MAX, u64::MAX / 2] {
            let v = hash_to_signed(seed);
            assert!(v >= -1.0 && v < 1.0, "out of range: {v} for seed {seed}");
        }
    }

    #[test]
    fn hash_to_signed_zero_maps_to_neg_one() {
        // Top 24 bits of 0 are all zero → 0 / 2^23 - 1 = -1.0.
        assert_eq!(hash_to_signed(0), -1.0);
    }

    // ── reject ────────────────────────────────────────────────────────────────

    #[test]
    fn reject_removes_parallel_component() {
        let v = DVec3::new(1.0, 2.0, 3.0);
        let axis = DVec3::Y;
        let r = reject(v, axis);
        assert!(r.dot(axis).abs() < 1e-12, "parallel component not removed");
        assert!((r - DVec3::new(1.0, 0.0, 3.0)).length() < 1e-12);
    }

    #[test]
    fn reject_perpendicular_is_identity() {
        let v = DVec3::X;
        let r = reject(v, DVec3::Y);
        assert!((r - v).length() < 1e-12);
    }

    #[test]
    fn reject_parallel_is_zero() {
        let r = reject(DVec3::Y * 5.0, DVec3::Y);
        assert!(r.length() < 1e-12);
    }

    // ── ray_aabb ──────────────────────────────────────────────────────────────

    #[test]
    fn ray_aabb_forward_hit() {
        let t = ray_aabb(
            DVec3::new(-5.0, 0.0, 0.0),
            DVec3::X,
            DVec3::splat(-1.0),
            DVec3::splat(1.0),
        );
        assert!((t.unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn ray_aabb_miss() {
        // Ray passes above the box in Y.
        assert!(ray_aabb(
            DVec3::new(-5.0, 2.0, 0.0),
            DVec3::X,
            DVec3::splat(-1.0),
            DVec3::splat(1.0),
        )
        .is_none());
    }

    #[test]
    fn ray_aabb_origin_inside() {
        // t_min is negative; the function returns t_max (exit distance).
        let t = ray_aabb(DVec3::ZERO, DVec3::X, DVec3::splat(-1.0), DVec3::splat(1.0));
        assert!((t.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ray_aabb_pointing_away() {
        // Both t_min and t_max are negative.
        assert!(ray_aabb(
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::X,
            DVec3::splat(-1.0),
            DVec3::splat(1.0),
        )
        .is_none());
    }

    #[test]
    fn ray_aabb_diagonal_hit() {
        // Ray from (-2,-2,0) along (1,1,0)/√2 enters at (-1,-1,0), t = √2.
        let dir = DVec3::new(1.0, 1.0, 0.0).normalize();
        let t = ray_aabb(
            DVec3::new(-2.0, -2.0, 0.0),
            dir,
            DVec3::splat(-1.0),
            DVec3::splat(1.0),
        );
        assert!((t.unwrap() - 2f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn ray_aabb_negative_direction_hit() {
        // Ray from (5,0,0) along (-1,0,0) hits box at x=1, t=4.
        let t = ray_aabb(
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::NEG_X,
            DVec3::splat(-1.0),
            DVec3::splat(1.0),
        );
        assert!((t.unwrap() - 4.0).abs() < 1e-10);
    }
}
