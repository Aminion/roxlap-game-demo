use legion::{system, world::SubWorld, *};

use crate::{components::cannon::Cannon, Dt};

#[system]
#[write_component(Cannon)]
pub fn cannon_cooldown(world: &mut SubWorld, #[resource] dt: &Dt) {
    let mut query = <&mut Cannon>::query();
    let cannon = query.iter_mut(world).next().expect("cannon missing");
    cannon.cooldown = (cannon.cooldown - dt.0).max(0.0);
}
