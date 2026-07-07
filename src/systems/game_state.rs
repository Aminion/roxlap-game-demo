use legion::system;

use crate::{components::energy::Energy, GameState};

#[system]
pub fn game_state_update(#[resource] energy: &Energy, #[resource] game_state: &mut GameState) {
    if matches!(*game_state, GameState::Playing) && energy.current <= 0.0 {
        *game_state = GameState::GameOver;
    }
}
