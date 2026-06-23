use legion::{world::SubWorld, *};

use crate::{
    components::{crystal::CrystalMarker, miner::Miner, newton_body::NewtonBody},
    Dt,
};

pub struct Energy {
    pub current: f64,
}

impl Energy {
    pub fn new(initial: f64) -> Self {
        Self { current: initial }
    }
}

/// Energy regenerated per crystal per second while within range.
pub const CRYSTAL_REGEN_RATE: f64 = 25.0;

/// Maximum distance from the miner at which a crystal provides regen.
const CRYSTAL_REGEN_DIST_SQ: f64 = 8.0 * 8.0;

fn compute_regen(current: f64, near_count: usize, dt: f64) -> f64 {
    if near_count == 0 {
        return current;
    }
    (current + CRYSTAL_REGEN_RATE * near_count as f64 * dt).min(ENERGY_MAX)
}

pub const ENERGY_MAX: f64 = 200.0;
pub const ENERGY_LOW: f64 = 30.0;
pub const ENERGY_MED: f64 = 90.0;
pub const SHOT_COST: f64 = 5.0;

#[system]
#[read_component(Miner)]
#[read_component(CrystalMarker)]
#[read_component(NewtonBody)]
pub fn energy(world: &SubWorld, #[resource] energy: &mut Energy, #[resource] dt: &Dt) {
    let miner_pos = {
        let mut q = <(&Miner, &NewtonBody)>::query();
        q.iter(world).next().expect("miner missing").1.pos
    };

    let mut crystal_q = <(&CrystalMarker, &NewtonBody)>::query();
    let near_count = crystal_q
        .iter(world)
        .filter(|(_, body)| body.pos.distance_squared(miner_pos) <= CRYSTAL_REGEN_DIST_SQ)
        .count();

    energy.current = compute_regen(energy.current, near_count, dt.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_regen_without_crystals() {
        assert_eq!(compute_regen(50.0, 0, 1.0), 50.0);
    }

    #[test]
    fn single_crystal_adds_correct_amount() {
        let result = compute_regen(0.0, 1, 1.0);
        assert!((result - CRYSTAL_REGEN_RATE).abs() < 1e-12);
    }

    #[test]
    fn two_crystals_add_double() {
        let result = compute_regen(0.0, 2, 1.0);
        assert!((result - 2.0 * CRYSTAL_REGEN_RATE).abs() < 1e-12);
    }

    #[test]
    fn regen_caps_at_energy_max() {
        let result = compute_regen(ENERGY_MAX - 1.0, 1, 1.0);
        assert_eq!(result, ENERGY_MAX);
    }
}
