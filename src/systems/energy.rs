use legion::{world::SubWorld, *};

use crate::{
    components::{asteroid::CrystalMarker, miner::Miner, newton_body::NewtonBody},
    Dt,
};

pub struct Energy {
    pub current: f64,
    pub max: f64,
}

impl Energy {
    pub fn new(max: f64) -> Self {
        Self { current: max, max }
    }
}

const CRYSTAL_REGEN_RATE: f64 = 20.0;
const CRYSTAL_REGEN_DIST: f64 = 3.0;

pub const SHOT_COST: f64 = 10.0;

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
    let near_crystal = crystal_q
        .iter(world)
        .any(|(_, body)| body.pos.distance(miner_pos) <= CRYSTAL_REGEN_DIST);

    if near_crystal {
        energy.current = (energy.current + CRYSTAL_REGEN_RATE * dt.0).min(energy.max);
    }
}
