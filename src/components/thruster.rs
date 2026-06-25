use glam::DVec3;

pub struct ThrusterBank {
    pub command: DVec3,
    pub linear_command: DVec3,
    pub max_rot_accel: f64,
    pub max_lin_accel: f64,
}

impl ThrusterBank {
    /// Build the bank, baking in sphere inertia `I = 2/5 · m · r²`.
    /// `max_rot_accel = 5·F / (m·r)` (two opposing nozzles per axis).
    /// `max_lin_accel = linear_force / mass`.
    pub fn new(mass: f64, radius: f64, force_per_thruster: f64, linear_force: f64) -> Self {
        Self {
            command: DVec3::ZERO,
            linear_command: DVec3::ZERO,
            max_rot_accel: (5.0 * force_per_thruster) / (mass * radius),
            max_lin_accel: linear_force / mass,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_rot_accel_formula() {
        let b = ThrusterBank::new(2.0, 3.0, 6.0, 10.0);
        let expected = 5.0 * 6.0 / (2.0 * 3.0);
        assert!((b.max_rot_accel - expected).abs() < 1e-10);
    }

    #[test]
    fn max_lin_accel_formula() {
        let b = ThrusterBank::new(4.0, 1.0, 1.0, 20.0);
        let expected = 20.0 / 4.0;
        assert!((b.max_lin_accel - expected).abs() < 1e-10);
    }
}
