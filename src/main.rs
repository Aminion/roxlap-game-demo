mod components;
mod systems;

use std::collections::HashSet;

use legion::{Resources, Schedule, World};
use roxlap_cavegen::pack_dense_grid_to_vxl;
use roxlap_core::Engine;
use roxlap_formats::{
    edit::{set_rect, MAXZDIM},
    vxl::Vxl,
};
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

const VSID: u32 = 32;

/// Z-coord of the (one-voxel-thick) ground plane. Voxlap is **z-down**:
/// small z is up, large z is down. `200` puts the floor near the
/// bottom of the voxlap z-range with ~200 voxels of empty air above
/// for the camera and the cube.
const GROUND_Z: i32 = 200;

/// Edge length of the demo cube, in voxels.
const CUBE_EDGE: i32 = 10;

/// Voxlap colour packing: `(brightness << 24) | (R << 16) | (G << 8) | B`.
/// `0x80` brightness is voxlap's neutral; the `update_lighting` bake
/// overwrites it with directional shading.
const GROUND_COL: u32 = 0x80_5a_a0_5a; // mossy green
const CUBE_COL: u32 = 0x80_c0_60_30; // warm orange

/// Walking speed, in voxels per second.
const MOVE_SPEED: f64 = 16.0;
/// Multiplier applied while `LCtrl` is held.
const FAST_MULT: f64 = 4.0;
/// Mouse sensitivity, in radians per pixel of cursor delta.
const MOUSE_SENS: f64 = 0.0025;
/// Pitch clamp — just shy of ±90° so the camera basis stays
/// well-conditioned (a straight-up view collapses `right × forward`).
const PITCH_LIMIT: f64 = 88.0_f64 * std::f64::consts::PI / 180.0;

fn build_world() -> Vxl {
    let vsid_u = VSID as usize;
    let maxz_u = MAXZDIM as usize;
    let cells = vsid_u * vsid_u * maxz_u;

    // Dense grid: 1-byte solid/air mask + matching u32 colour, stored
    // in `(y, x, z)` order (the layout `pack_dense_grid_to_vxl`
    // expects). Start every voxel as air, then stamp the ground row.
    let mut mask = vec![0u8; cells];
    let mut colour = vec![0u32; cells];
    let idx = |x: usize, y: usize, z: usize| -> usize { (y * vsid_u + x) * maxz_u + z };
    for y in 0..vsid_u {
        for x in 0..vsid_u {
            let i = idx(x, y, GROUND_Z as usize);
            mask[i] = 1;
            colour[i] = GROUND_COL;
        }
    }
    let mut world = pack_dense_grid_to_vxl(&mask, &colour, VSID);

    // Reserve a slab pool large enough for the cube edit. The edit
    // API allocates new slab records into this pool; without it
    // `set_rect` panics. 64 KB is overkill for one 10³ cube but fits
    // comfortably and gives downstream tinkering some headroom.
    world.reserve_edit_capacity(64 * 1024);

    // Place the cube centred on the world's XY footprint, sitting
    // directly on the ground (top of cube `CUBE_EDGE` voxels above
    // GROUND_Z, i.e. lower z because voxlap is z-down).
    let cx = (VSID as i32) / 2;
    let cy = (VSID as i32) / 2;
    let half = CUBE_EDGE / 2;
    let lo = [cx - half, cy - half, GROUND_Z - CUBE_EDGE];
    let hi = [cx + half - 1, cy + half - 1, GROUND_Z - 1];
    set_rect(&mut world, lo, hi, Some(CUBE_COL));

    world
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

    engine.set_side_shades(15, 15, 15, 15, 15, 15);
    engine.set_lightmode(1);
    resources.insert(engine);
    resources.insert(canvas_resources);
    resources.insert(HashSet::<PlayerInput>::new());

    let vxl = build_world();
    resources.insert(vxl);

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
