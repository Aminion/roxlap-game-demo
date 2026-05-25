mod systems;

use std::collections::HashSet;

use legion::{Resources, Schedule, World};
use roxlap_core::Engine;
use roxlap_formats::vxl::Vxl;
use sdl2::{
    event::Event,
    keyboard::Scancode,
    mixer::InitFlag,
    render::{Canvas, TextureCreator, WindowCanvas},
    video::{Window, WindowContext},
    EventPump,
};

use crate::systems::render::render_system;

const INITIAL_WINDOW_WIDTH: u32 = 1280;
const INITIAL_WINDOW_HEIGHT: u32 = 720;

pub struct CanvasResources {
    pub canvas: Canvas<Window>,
    pub texture_creator: TextureCreator<WindowContext>,
}

fn initialize() -> Result<(WindowCanvas, EventPump), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let _audio = sdl_context.audio()?;

    let _mixer_context =
        sdl2::mixer::init(InitFlag::MP3 | InitFlag::FLAC | InitFlag::MOD | InitFlag::OGG)?;
    sdl2::mixer::allocate_channels(20);

    let window = video_subsystem
        .window(
            "ROXLAP GAME DEMO",
            INITIAL_WINDOW_WIDTH,
            INITIAL_WINDOW_HEIGHT,
        )
        .resizable()
        .position_centered()
        .build()
        .expect("could not initialize video subsystem");

    let mut canvas = window
        .into_canvas()
        .accelerated()
        .present_vsync()
        .build()
        .expect("could not make a canvas");

    canvas.present();

    let event_pump = sdl_context.event_pump().unwrap();

    Ok((canvas, event_pump))
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub enum PlayerInput {
    PitchCW,
    PitchCCW,
    YawCW,
    YawCCW,
    RollCW,
    RollCCW,
    IncTrust,
    DecTrust,
}

impl PlayerInput {
    pub fn from_scancode(scancode: Scancode) -> Option<Self> {
        match scancode {
            Scancode::A => Some(PlayerInput::RollCCW),
            Scancode::D => Some(PlayerInput::RollCW),
            Scancode::W => Some(PlayerInput::PitchCCW),
            Scancode::S => Some(PlayerInput::PitchCW),
            Scancode::Q => Some(PlayerInput::YawCCW),
            Scancode::E => Some(PlayerInput::YawCW),
            Scancode::LShift => Some(PlayerInput::IncTrust),
            Scancode::LCtrl => Some(PlayerInput::DecTrust),
            _ => None,
        }
    }
}

fn initial_resources(canvas: Canvas<Window>, world: &World) -> Resources {
    let mut resources = Resources::default();
    let texture_creator = canvas.texture_creator();

    let canvas_resources = CanvasResources {
        canvas,
        texture_creator,
    };
    let mut engine = Engine::new();

    let cube_color: u32 = 0x00FF00FF; // ARGB Magenta
    engine.set_sky_color(0x00224466);
    resources.insert(engine);
    resources.insert(canvas_resources);
    resources.insert(HashSet::<PlayerInput>::new());

    resources
}

fn main() {
    std::env::set_var("RUST_LOG", "info");
    std::env::set_var("RUST_BACKTRACE", "1");
    env_logger::init();
    let (canvas, mut event_pump) = initialize().unwrap();

    let mut schedule = Schedule::builder()
        .add_thread_local(render_system())
        .build();
    let mut world = World::default();
    let mut resources = initial_resources(canvas, &mut world);

    'running: loop {
        for event in event_pump.poll_iter() {
            let mut pinput = resources.get_mut::<HashSet<PlayerInput>>().unwrap();
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    scancode: Some(Scancode::Escape),
                    ..
                } => break 'running,
                Event::KeyDown {
                    scancode: Some(code),
                    ..
                } => {
                    let insertion = PlayerInput::from_scancode(code);
                    if let Some(player_input) = insertion {
                        pinput.insert(player_input);
                    }
                }
                Event::KeyUp {
                    scancode: Some(code),
                    ..
                } => {
                    let deletion = PlayerInput::from_scancode(code);
                    if let Some(player_input) = deletion {
                        pinput.remove(&player_input);
                    }
                }
                _ => {}
            }
        }

        schedule.execute(&mut world, &mut resources);
    }
}
