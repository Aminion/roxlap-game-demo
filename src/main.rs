mod components;
mod fonts;
mod input;
mod math;
mod systems;
mod world;

use std::collections::HashSet;
use std::time::Instant;

use glam::{DVec3, Vec2};
use legion::{Resources, Schedule, World};
use roxlap_core::{rasterizer::ScratchPool, update_lighting, Engine};
use roxlap_formats::edit::MAXZDIM;
use sdl2::{
    event::{Event, WindowEvent},
    keyboard::Scancode,
    mixer::InitFlag,
    pixels::PixelFormatEnum,
    render::{Canvas, Texture, TextureCreator, WindowCanvas},
    video::{Window, WindowContext},
    EventPump,
};

use crate::fonts::FontRenderer;
use crate::input::PlayerInput;
use crate::systems::{
    autopilot::autopilot_system,
    camera::camera_update_system,
    miner_input::miner_input_system,
    newton_body::newton_body_system,
    performance_info::{update_info_system, PerformanceInfo},
    render::render_system,
    thruster::thruster_system,
};
use crate::world::{
    build_cube_vxl, build_world, miner_initial_forward, populate_world, Worlds, CUBE_VXL_VSID, VSID,
};

const INITIAL_WINDOW_WIDTH: u32 = 1280;
const INITIAL_WINDOW_HEIGHT: u32 = 720;

/// Window dimensions plus the world-space target direction for the rotation autopilot.
/// Combined into one resource so the render system stays within Legion's 8-resource limit.
pub struct ScreenState {
    pub width: u32,
    pub height: u32,
    /// World-space unit vector: where the autopilot should point the ship's nose.
    pub target_dir: DVec3,
}

pub struct Dt(pub f64);

/// Accumulated mouse motion for the current frame, reset before each event poll.
pub type MouseDelta = Vec2;

pub struct FrameTimer(pub Instant);

pub struct CanvasResources {
    pub canvas: Canvas<Window>,
    pub texture_creator: TextureCreator<WindowContext>,
}

/// Long-lived streaming texture for the 3-D framebuffer.
/// Lives outside CanvasResources so the canvas and the texture can be
/// borrowed independently in the render system.
pub struct RenderTexture(pub Texture);

/// Reusable per-frame scratch: opticast pool, pixel buffer, depth buffer.
/// Recreated whenever the window is resized.
pub struct RenderBuffers {
    pub pool: ScratchPool,
    pub framebuffer: Vec<u32>,
    pub zbuffer: Vec<f32>,
    /// Secondary buffers for the cube rendering pass.
    pub cube_fb: Vec<u32>,
    pub cube_zb: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl RenderBuffers {
    pub fn new(width: u32, height: u32, vsid: u32) -> Self {
        let n = (width * height) as usize;
        // Pool must be large enough for both the ground (vsid) and the cube VXL.
        let pool_vsid = vsid.max(CUBE_VXL_VSID);
        let mut pool = ScratchPool::new(width, height, pool_vsid);
        pool.set_treat_z_max_as_air(true);
        Self {
            pool,
            framebuffer: vec![0u32; n],
            zbuffer: vec![0.0f32; n],
            cube_fb: vec![0u32; n],
            cube_zb: vec![0.0f32; n],
            width,
            height,
        }
    }
}

fn initialize() -> Result<(WindowCanvas, EventPump), String> {
    let sdl_context = sdl2::init()?;
    sdl2::hint::set("SDL_RENDER_SCALE_QUALITY", "best");
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

    sdl_context.mouse().set_relative_mouse_mode(true);

    let event_pump = sdl_context.event_pump()?;

    Ok((canvas, event_pump))
}

fn initial_resources(canvas: Canvas<Window>) -> Resources {
    let mut resources = Resources::default();
    let texture_creator = canvas.texture_creator();

    // Create the streaming texture at the initial window size; the render system
    // recreates it whenever WindowSize changes.
    let render_texture = RenderTexture(
        texture_creator
            .create_texture_streaming(
                PixelFormatEnum::ARGB8888,
                INITIAL_WINDOW_WIDTH / 2,
                INITIAL_WINDOW_HEIGHT / 2,
            )
            .expect("failed to create render texture"),
    );

    let canvas_resources = CanvasResources {
        canvas,
        texture_creator,
    };

    let mut engine = Engine::new();
    engine.set_side_shades(15, 0, 8, 8, 8, 8);
    engine.set_lightmode(1);

    let mut vxl = build_world();
    update_lighting(
        &mut vxl.data,
        &vxl.column_offset,
        vxl.vsid,
        0,
        0,
        0,
        vxl.vsid as i32,
        vxl.vsid as i32,
        MAXZDIM,
        engine.lightmode(),
        engine.lights(),
    );

    let mut cube_vxl = build_cube_vxl();
    update_lighting(
        &mut cube_vxl.data,
        &cube_vxl.column_offset,
        cube_vxl.vsid,
        0,
        0,
        0,
        cube_vxl.vsid as i32,
        cube_vxl.vsid as i32,
        MAXZDIM,
        engine.lightmode(),
        engine.lights(),
    );

    resources.insert(engine);
    resources.insert(canvas_resources);
    resources.insert(render_texture);
    resources.insert(ScreenState {
        width: INITIAL_WINDOW_WIDTH,
        height: INITIAL_WINDOW_HEIGHT,
        target_dir: miner_initial_forward(),
    });
    resources.insert(Vec2::ZERO);
    resources.insert(RenderBuffers::new(
        INITIAL_WINDOW_WIDTH / 2,
        INITIAL_WINDOW_HEIGHT / 2,
        VSID,
    ));
    resources.insert(HashSet::<PlayerInput>::new());
    resources.insert(Worlds {
        base: vxl,
        cube: cube_vxl,
    });
    resources.insert(FrameTimer(Instant::now()));
    // Overwritten at the top of every game-loop iteration before any system runs.
    resources.insert(Dt(0.0));
    resources.insert(FontRenderer::new());
    resources.insert(PerformanceInfo::new());

    resources
}

fn build_schedule() -> Schedule {
    Schedule::builder()
        .add_system(update_info_system())
        .add_system(miner_input_system())
        .add_system(camera_update_system())
        .add_system(autopilot_system())
        .add_system(thruster_system())
        .add_system(newton_body_system())
        .add_thread_local(render_system())
        .build()
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let (canvas, mut event_pump) = initialize().unwrap();

    let mut schedule = build_schedule();
    let mut world = World::default();
    let mut resources = initial_resources(canvas);

    populate_world(&mut world);

    'running: loop {
        {
            let mut frame_timer = resources.get_mut::<FrameTimer>().unwrap();
            let mut dt = resources.get_mut::<Dt>().unwrap();
            dt.0 = frame_timer.0.elapsed().as_secs_f64();
            frame_timer.0 = Instant::now();
        }

        {
            *resources.get_mut::<MouseDelta>().unwrap() = Vec2::ZERO;
        }

        for event in event_pump.poll_iter() {
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
                    if let Some(input) = PlayerInput::from_scancode(code) {
                        resources
                            .get_mut::<HashSet<PlayerInput>>()
                            .unwrap()
                            .insert(input);
                    }
                }
                Event::KeyUp {
                    scancode: Some(code),
                    ..
                } => {
                    if let Some(input) = PlayerInput::from_scancode(code) {
                        resources
                            .get_mut::<HashSet<PlayerInput>>()
                            .unwrap()
                            .remove(&input);
                    }
                }
                Event::MouseMotion { xrel, yrel, .. } => {
                    *resources.get_mut::<MouseDelta>().unwrap() +=
                        Vec2::new(xrel as f32, yrel as f32);
                }
                Event::Window {
                    win_event: WindowEvent::Resized(x, y),
                    ..
                } => {
                    let mut ss = resources.get_mut::<ScreenState>().unwrap();
                    ss.width = x.max(1) as u32;
                    ss.height = y.max(1) as u32;
                }
                _ => {}
            }
        }

        schedule.execute(&mut world, &mut resources);
    }
}
