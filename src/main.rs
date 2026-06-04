mod components;
mod fonts;
mod systems;

use std::collections::HashSet;
use std::time::Instant;

use glam::{DMat3, DQuat, DVec3};
use legion::{Resources, Schedule, World};
use roxlap_cavegen::pack_dense_grid_to_vxl;
use roxlap_core::{rasterizer::ScratchPool, update_lighting, Camera, Engine};
use roxlap_formats::{edit::MAXZDIM, vxl::Vxl};
use sdl2::{
    event::{Event, WindowEvent},
    keyboard::Scancode,
    mixer::InitFlag,
    pixels::PixelFormatEnum,
    render::{Canvas, Texture, TextureCreator, WindowCanvas},
    video::{Window, WindowContext},
    EventPump,
};

use crate::components::{
    camera::CameraComponent, cube_marker::CubeMarker, miner::Miner, newton_body::NewtonBody,
};
use crate::fonts::FontRenderer;
use crate::systems::{
    camera::camera_update_system,
    miner_input::miner_input_system,
    newton_body::newton_body_system,
    performance_info::{update_info_system, PerformanceInfo},
    render::render_system,
};

const INITIAL_WINDOW_WIDTH: u32 = 1280;
const INITIAL_WINDOW_HEIGHT: u32 = 720;

/// Current window / render resolution, updated by the resize event handler.
pub struct WindowSize(pub u32, pub u32);

pub struct Dt(pub f64);

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
        Self {
            pool: ScratchPool::new(width, height, pool_vsid),
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

const VSID: u32 = 128;

/// Z-coord of the (one-voxel-thick) ground plane. Voxlap is **z-down**:
/// small z is up, large z is down. `200` puts the floor near the
/// bottom of the voxlap z-range with ~200 voxels of empty air above
/// for the camera and the cube.
const GROUND_Z: i32 = 200;

/// Cube VXL dimensions. Separate from the ground world (VSID=32).
pub const CUBE_VXL_VSID: u32 = 128;
pub const CUBE_VXL_EDGE: i32 = 64;

/// Voxlap colour packing: `(brightness << 24) | (R << 16) | (G << 8) | B`.
/// `0x80` brightness is voxlap's neutral; the `update_lighting` bake
/// overwrites it with directional shading.
const GROUND_COL: u32 = 0x80_5a_a0_5a; // mossy green
const CUBE_COL: u32 = 0x80_c0_60_30; // warm orange

fn build_world() -> Vxl {
    let vsid_u = VSID as usize;
    let maxz_u = MAXZDIM as usize;
    let cells = vsid_u * vsid_u * maxz_u;

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
    pack_dense_grid_to_vxl(&mask, &colour, VSID)
}

fn build_cube_vxl() -> Vxl {
    let vsid = CUBE_VXL_VSID as usize;
    let maxz_u = MAXZDIM as usize;
    let cells = vsid * vsid * maxz_u;

    let mut mask = vec![0u8; cells];
    let mut colour = vec![0u32; cells];
    let idx = |x: usize, y: usize, z: usize| -> usize { (y * vsid + x) * maxz_u + z };

    let half = (CUBE_VXL_EDGE / 2) as usize;
    let center = vsid / 2; // 64
    let lo = center - half; // 32
    let hi = center + half; // 96 (exclusive)

    for y in lo..hi {
        for x in lo..hi {
            for z in lo..hi {
                mask[idx(x, y, z)] = 1;
                colour[idx(x, y, z)] = CUBE_COL;
            }
        }
    }
    pack_dense_grid_to_vxl(&mask, &colour, CUBE_VXL_VSID)
}

/// Holds both the static ground world and the pre-lit cube VXL.
/// Bundled into one resource to stay within Legion's 8-resource-per-system limit.
pub struct Worlds {
    pub base: Vxl,
    pub cube: Vxl,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum PlayerInput {
    PitchCW,
    PitchCCW,
    YawCW,
    YawCCW,
    RollCW,
    RollCCW,
    IncTrust,
    DecTrust,
    Damp,
}

impl PlayerInput {
    pub fn from_scancode(scancode: Scancode) -> Option<Self> {
        match scancode {
            Scancode::A => Some(PlayerInput::YawCW),
            Scancode::D => Some(PlayerInput::YawCCW),
            Scancode::W => Some(PlayerInput::PitchCW),
            Scancode::S => Some(PlayerInput::PitchCCW),
            Scancode::Q => Some(PlayerInput::RollCCW),
            Scancode::E => Some(PlayerInput::RollCW),
            Scancode::LShift => Some(PlayerInput::IncTrust),
            Scancode::LCtrl => Some(PlayerInput::DecTrust),
            Scancode::Tab => Some(PlayerInput::Damp),
            _ => None,
        }
    }
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
                INITIAL_WINDOW_WIDTH,
                INITIAL_WINDOW_HEIGHT,
            )
            .expect("failed to create render texture"),
    );

    let canvas_resources = CanvasResources {
        canvas,
        texture_creator,
    };

    let mut engine = Engine::new();
    engine.set_side_shades(15, 15, 15, 15, 15, 15);
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
    resources.insert(WindowSize(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT));
    resources.insert(RenderBuffers::new(
        INITIAL_WINDOW_WIDTH,
        INITIAL_WINDOW_HEIGHT,
        VSID,
    ));
    resources.insert(HashSet::<PlayerInput>::new());
    resources.insert(Worlds { base: vxl, cube: cube_vxl });
    resources.insert(FrameTimer(Instant::now()));
    resources.insert(Dt(0.0));
    resources.insert(FontRenderer::new());
    resources.insert(PerformanceInfo::new());

    resources
}

fn main() {
    std::env::set_var("RUST_LOG", "info");
    std::env::set_var("RUST_BACKTRACE", "1");
    env_logger::init();
    let (canvas, mut event_pump) = initialize().unwrap();

    let mut schedule = Schedule::builder()
        .add_system(update_info_system())
        .add_system(miner_input_system())
        .add_system(newton_body_system())
        .add_system(camera_update_system())
        .add_thread_local(render_system())
        .build();
    let mut world = World::default();
    let mut resources = initial_resources(canvas);

    // Spawn the player entity.  Initial orientation matches the old hardcoded
    // demo camera: looking toward +X (yaw=0), pitched 0.15 rad nose-down.
    //
    // Body-local conventions: -Z=forward, +X=right, +Y=up.
    // We build a rotation matrix whose columns describe where each body axis
    // ends up in voxlap world space, then convert to a quaternion.
    {
        // 46° nose-down so the camera looks at the ground beneath the cube.
        let pitch: f64 = 0.8;
        let (sp, cp) = (pitch.sin(), pitch.cos());
        // where body +X goes  →  world +Y  (right wing = horizontal right)
        // where body +Y goes  →  world (-sp, 0, cp)  (top = voxlap-sky direction)
        // where body +Z goes  →  world (-cp, 0, -sp)  (tail = -forward)
        let orientation = DQuat::from_mat3(&DMat3::from_cols(
            DVec3::Y,
            DVec3::new(-sp, 0.0, cp),
            DVec3::new(-cp, 0.0, -sp),
        ))
        .normalize();

        let cx = f64::from(VSID) * 0.5;
        let cy = f64::from(VSID) * 0.5;
        // 100 voxels above the ground, 70 to the left of world centre.
        // At pitch=0.8 the forward ray hits the ground within the 128-wide world.
        let cz = f64::from(GROUND_Z) - 100.0;
        let pos = DVec3::new(cx - 70.0, cy, cz);
        let fwd = orientation * DVec3::NEG_Z;
        let right = orientation * DVec3::X;
        let up = orientation * DVec3::Y;
        world.push((
            Miner,
            NewtonBody {
                mass: 1.0,
                pos,
                vel: DVec3::ZERO,
                orientation,
                angular_vel: DVec3::ZERO,
            },
            CameraComponent(Camera {
                pos: pos.to_array(),
                forward: fwd.to_array(),
                right: right.to_array(),
                down: (-up).to_array(),
            }),
        ));
    }

    // Cube entity: spins in place above the ground plane.
    // pos = world-space center of the cube (sits on ground at z=GROUND_Z).
    {
        let cube_center = DVec3::new(
            f64::from(VSID) / 2.0,
            f64::from(VSID) / 2.0,
            f64::from(GROUND_Z) - f64::from(CUBE_VXL_EDGE) / 2.0 - 15.0,
        );
        world.push((
            CubeMarker,
            NewtonBody {
                mass: 1.0,
                pos: cube_center,
                vel: DVec3::ZERO,
                orientation: DQuat::IDENTITY,
                angular_vel: DVec3::new(0.3, 0.2, 0.1),
            },
        ));
    }

    'running: loop {
        {
            let mut frame_timer = resources.get_mut::<FrameTimer>().unwrap();
            let mut dt = resources.get_mut::<Dt>().unwrap();
            dt.0 = frame_timer.0.elapsed().as_secs_f64();
            frame_timer.0 = Instant::now();
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
                Event::Window {
                    win_event: WindowEvent::Resized(x, y),
                    ..
                } => {
                    let mut ws = resources.get_mut::<WindowSize>().unwrap();
                    ws.0 = x.max(1) as u32;
                    ws.1 = y.max(1) as u32;
                }
                _ => {}
            }
        }

        schedule.execute(&mut world, &mut resources);
    }
}
