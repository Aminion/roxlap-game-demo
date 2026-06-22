use crate::components::{newton_body::NewtonBody, thruster::ThrusterBank};
use glam::{DQuat, DVec3};

pub fn make_body() -> NewtonBody {
    NewtonBody {
        mass: 1.0,
        pos: DVec3::ZERO,
        vel: DVec3::ZERO,
        orientation: DQuat::IDENTITY,
        angular_vel: DVec3::ZERO,
    }
}

pub fn make_bank() -> ThrusterBank {
    ThrusterBank::new(1.0, 1.0, 0.6, 5.0)
}
