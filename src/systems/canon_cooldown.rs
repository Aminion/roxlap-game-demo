use legion::{system, world::SubWorld, *};

use crate::{components::canon::Canon, Dt};

#[system]
#[write_component(Canon)]
pub fn canon_cooldown(world: &mut SubWorld, #[resource] dt: &Dt) {
    let mut query = <&mut Canon>::query();
    let canon = query.iter_mut(world).next().expect("canon missing");
    canon.cooldown = (canon.cooldown - dt.0).max(0.0);
}
