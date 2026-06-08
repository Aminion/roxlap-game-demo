use sdl2::keyboard::Scancode;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum PlayerInput {
    RollCW,
    RollCCW,
    ThrustUp,
    ThrustDown,
    ThrustLeft,
    ThrustRight,
    ThrustForward,
    ThrustBackward,
    Damping,
}

impl PlayerInput {
    pub fn from_scancode(scancode: Scancode) -> Option<Self> {
        match scancode {
            Scancode::Q => Some(PlayerInput::RollCCW),
            Scancode::E => Some(PlayerInput::RollCW),
            Scancode::W => Some(PlayerInput::ThrustUp),
            Scancode::S => Some(PlayerInput::ThrustDown),
            Scancode::A => Some(PlayerInput::ThrustLeft),
            Scancode::D => Some(PlayerInput::ThrustRight),
            Scancode::LShift => Some(PlayerInput::ThrustForward),
            Scancode::Space => Some(PlayerInput::ThrustBackward),
            Scancode::Tab => Some(PlayerInput::Damping),
            _ => None,
        }
    }
}
