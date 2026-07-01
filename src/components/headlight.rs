use roxlap_render::DirectionalLight;

#[derive(Clone, Copy, Debug)]
pub struct Headlight(pub Option<DirectionalLight>);
