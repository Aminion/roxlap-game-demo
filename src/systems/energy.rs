use legion::{world::SubWorld, *};

use crate::{
    components::{asteroid::CrystalMarker, miner::Miner, newton_body::NewtonBody},
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

/// Energy cap — prevents unlimited accumulation near crystal clusters.
pub const MAX_ENERGY: f64 = 300.0;

/// Energy regenerated per crystal per second while within range.
const CRYSTAL_REGEN_RATE: f64 = 25.0;

/// Maximum distance from the miner at which a crystal provides regen.
const CRYSTAL_REGEN_DIST: f64 = 8.0;

pub const SHOT_COST: f64 = 5.0;

#[system]
#[read_component(Miner)]
#[read_component(CrystalMarker)]
#[read_component(NewtonBody)]
pub fn energy(world: &SubWorld, #[resource] energy: &mut Energy, #[resource] dt: &Dt) {
    let miner_pos = {
        let mut q = <(&Miner, &NewtonBody)>::query();
        q.iter(world).next().map(|(_, body)| body.pos)
    };

    let Some(miner_pos) = miner_pos else { return };

    let mut crystal_q = <(&CrystalMarker, &NewtonBody)>::query();
    let near_count = crystal_q
        .iter(world)
        .filter(|(_, body)| body.pos.distance(miner_pos) <= CRYSTAL_REGEN_DIST)
        .count();

    if near_count > 0 {
        energy.current =
            (energy.current + CRYSTAL_REGEN_RATE * near_count as f64 * dt.0).min(MAX_ENERGY);
    }
}
