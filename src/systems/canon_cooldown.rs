use legion::{system, world::SubWorld, *};

use crate::{components::canon::Canon, Dt};

#[system]
#[write_component(Canon)]
pub fn canon_cooldown(world: &mut SubWorld, #[resource] dt: &Dt) {
    let mut query = <&mut Canon>::query();
    for canon in query.iter_mut(world) {
        canon.cooldown = (canon.cooldown - dt.0).max(0.0);
    }
}
