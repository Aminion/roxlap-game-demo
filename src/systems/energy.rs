pub struct Energy {
    pub current: f64,
}

impl Energy {
    pub fn new(initial: f64) -> Self {
        Self { current: initial }
    }
}

pub const ENERGY_LOW: f64 = 30.0;
pub const ENERGY_MED: f64 = 90.0;
pub const SHOT_COST: f64 = 5.0;
