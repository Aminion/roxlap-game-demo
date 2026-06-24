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
pub fn ray_aabb(ray_origin: DVec3, ray_dir: DVec3, center: DVec3, half: f64) -> Option<f64> {
    let inv = ray_dir.recip();
    let t1 = (center - half - ray_origin) * inv;
    let t2 = (center + half - ray_origin) * inv;
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
}
