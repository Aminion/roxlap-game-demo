use legion::system;
use roxlap_render::{ParticleSystem, SceneRenderer};

use crate::Dt;

pub const PARTICLE_MODEL_DIM: f32 = 3.0;

#[system]
pub fn particle(
    #[resource] particle_sys: &mut ParticleSystem,
    #[resource] renderer: &mut SceneRenderer,
    #[resource] dt: &Dt,
) {
    particle_sys.tick(renderer, dt.0);
}
