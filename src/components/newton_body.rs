use glam::{DQuat, DVec3};

use crate::Dt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewtonBody {
    pub mass: f64,
    pub pos: DVec3,
    pub vel: DVec3,
    pub orientation: DQuat,
    pub angular_vel: DVec3,
}

impl NewtonBody {
    pub fn integrate_rotation(&mut self, dt: &Dt) {
        self.orientation =
            (DQuat::from_scaled_axis(self.angular_vel * dt.0) * self.orientation).normalize();
    }
}

#[cfg(test)]
mod tests {
    use super::NewtonBody;
    use crate::{test_utils::make_body, Dt};
    use glam::{DQuat, DVec3};
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn zero_angular_vel_leaves_orientation_unchanged() {
        let mut body = make_body();
        body.integrate_rotation(&Dt(1.0));
        let dot = body.orientation.dot(DQuat::IDENTITY).abs();
        assert!(dot > 1.0 - 1e-12, "identity orientation must not change");
    }

    #[test]
    fn dt_zero_leaves_orientation_unchanged() {
        let mut body = make_body();
        body.angular_vel = DVec3::new(1.0, 2.0, 3.0);
        let before = body.orientation;
        body.integrate_rotation(&Dt(0.0));
        let dot = body.orientation.dot(before).abs();
        assert!(dot > 1.0 - 1e-12, "dt=0 must not change orientation");
    }

    #[test]
    fn single_axis_rotation_integrates_correctly() {
        // 1 rad/s around X for π/2 seconds → should equal from_rotation_x(π/2).
        let mut body = make_body();
        body.angular_vel = DVec3::X;
        body.integrate_rotation(&Dt(FRAC_PI_2));
        let expected = DQuat::from_rotation_x(FRAC_PI_2);
        let dot = body.orientation.dot(expected).abs();
        assert!(dot > 1.0 - 1e-9, "single-step rotation_x(π/2): dot={dot}");
    }

    #[test]
    fn stays_normalized_after_many_steps() {
        let mut body = make_body();
        body.angular_vel = DVec3::new(1.0, 0.5, -0.3);
        let dt = Dt(1.0 / 60.0);
        for _ in 0..600 {
            body.integrate_rotation(&dt);
        }
        let len = body.orientation.length();
        assert!(
            (len - 1.0).abs() < 1e-9,
            "|q| drifted to {len} after 600 steps"
        );
    }
}
