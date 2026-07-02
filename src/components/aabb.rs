use glam::DVec3;

#[derive(Clone)]
pub struct Aabb {
    pub min: DVec3,
    pub max: DVec3,
}

impl Aabb {
    /// Inverted-infinity sentinel: fails all intersection tests until recomputed.
    pub fn empty() -> Self {
        Self {
            min: DVec3::splat(f64::INFINITY),
            max: DVec3::splat(f64::NEG_INFINITY),
        }
    }

    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min.cmple(other.max).all() && self.max.cmpge(other.min).all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aabb(min: [f64; 3], max: [f64; 3]) -> Aabb {
        Aabb {
            min: DVec3::from(min),
            max: DVec3::from(max),
        }
    }

    #[test]
    fn overlaps_separated() {
        let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = aabb([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]);
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn overlaps_touching_face() {
        // cmple / cmpge use <=/>= so touching counts as overlapping.
        let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = aabb([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn overlaps_proper_overlap() {
        let a = aabb([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = aabb([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn overlaps_nested() {
        let outer = aabb([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
        let inner = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert!(outer.overlaps(&inner));
        assert!(inner.overlaps(&outer));
    }

    #[test]
    fn empty_sentinel_fails_all_overlaps() {
        // The inverted-infinity sentinel must reject every possible AABB.
        let e = Aabb::empty();
        let huge = aabb([-1e15, -1e15, -1e15], [1e15, 1e15, 1e15]);
        assert!(!e.overlaps(&huge));
        assert!(!huge.overlaps(&e));
        assert!(!e.overlaps(&e));
    }
}
