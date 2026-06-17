mod components;
mod generation;
mod input;
mod math;
mod systems;
mod world;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, IVec3, Vec2};
use legion::{Entity, Resources, Schedule, World};
use raw_window_handle::{
    DisplayHandle, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    WindowHandle,
};
use roxlap_gpu::{
    GpuRenderer, GpuRendererSettings, GpuSceneResident, SceneUpload, SpriteModelRegistry,
};
use sdl2::{
    event::{Event, WindowEvent},
    keyboard::Scancode,
    mixer::InitFlag,
    video::Window,
    EventPump,
};

use crate::input::PlayerInput;
use crate::systems::{
    autopilot::autopilot_system,
    camera::camera_update_system,
    miner_input::miner_input_system,
    newton_body::newton_body_system,
    performance_info::{update_info_system, PerformanceInfo},
    presence_position::presence_position_update_system,
    render::render_system,
    thruster::thruster_system,
};
use crate::world::{generate_star_sky, miner_initial_forward, populate_world};

const INITIAL_WINDOW_WIDTH: u32 = 1280;
const INITIAL_WINDOW_HEIGHT: u32 = 720;
const WORLD_SEED: u64 = 42;

pub struct ScreenState {
    pub width: u32,
    pub height: u32,
    pub fov_y_rad: f32,
}

/// World-space unit vector: where the autopilot should point the ship's nose.
pub struct AutopilotTarget(pub DVec3);

pub struct Dt(pub f64);

/// Accumulated mouse motion for the current frame, reset before each event poll.
pub type MouseDelta = Vec2;

pub struct FrameTimer(pub Instant);

// --- GPU resources ---

/// GPU-resident voxel scene (base world grid).
pub struct GpuWorldData {
    pub scene: GpuSceneResident,
}

/// CPU-side sprite model registry, kept alive so future edits (destruction)
/// can call `gpu.update_sprite_model(&sprite_data.registry, chain_id)`.
pub struct SpriteData {
    pub registry: SpriteModelRegistry,
}

/// Set of chunk coordinates (in chunk-space) that have already been visited and populated.
pub struct VisitedChunks(pub HashSet<IVec3>);

/// Set of asteroid entity IDs currently loaded within the presence area.
pub struct LoadedAsteroids(pub HashSet<Entity>);

/// Seed for all procedural world generation (chunk density noise, asteroid properties).
pub struct WorldSeed(pub u64);

// --- SDL2 window handle wrapper for wgpu ---

/// Snapshot of an SDL2 window's raw handles for wgpu surface creation.
///
/// The handles are captured once at construction and returned by value on every
/// call. This avoids re-querying SDL2's WM info per frame and matches the
/// pattern used by the upstream `roxlap-sdl-demo` reference.
///
/// # Safety
/// Holds only `Copy` raw handles (no SDL state), so `Send + Sync` is sound as
/// long as the backing SDL window outlives this adapter.
struct SdlWindowHandle {
    window: RawWindowHandle,
    display: RawDisplayHandle,
}

unsafe impl Send for SdlWindowHandle {}
unsafe impl Sync for SdlWindowHandle {}

impl HasWindowHandle for SdlWindowHandle {
    fn window_handle(&self) -> Result<WindowHandle<'_>, raw_window_handle::HandleError> {
        Ok(unsafe { WindowHandle::borrow_raw(self.window) })
    }
}

impl HasDisplayHandle for SdlWindowHandle {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(unsafe { DisplayHandle::borrow_raw(self.display) })
    }
}

fn initialize() -> Result<(Window, EventPump), String> {
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
        .fullscreen()
        .build()
        .expect("could not initialize video subsystem");

    sdl_context.mouse().set_relative_mouse_mode(true);

    let event_pump = sdl_context.event_pump()?;

    Ok((window, event_pump))
}

fn initial_resources(handle: Arc<SdlWindowHandle>) -> Resources {
    let mut resources = Resources::default();

    let mut gpu = GpuRenderer::new_blocking(
        handle,
        (INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT),
        GpuRendererSettings {
            uncapped_present: true,
            ..GpuRendererSettings::default()
        },
    )
    .expect("GPU init failed — no Vulkan/Metal/DX12 adapter?");
    let (sky_pixels, sky_w, sky_h) = generate_star_sky(WORLD_SEED);
    gpu.set_sky_panorama(&sky_pixels, sky_w, sky_h);

    let gpu_world = GpuWorldData {
        scene: GpuSceneResident::upload(gpu.device(), &SceneUpload { grids: vec![] }),
    };

    let sprite_registry = SpriteModelRegistry::new();

    resources.insert(ScreenState {
        width: INITIAL_WINDOW_WIDTH,
        height: INITIAL_WINDOW_HEIGHT,
        fov_y_rad: fov_y(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT),
    });
    resources.insert(AutopilotTarget(miner_initial_forward()));
    resources.insert(Vec2::ZERO);
    resources.insert(HashSet::<PlayerInput>::new());
    resources.insert(FrameTimer(Instant::now()));
    resources.insert(Dt(0.0));
    resources.insert(egui::Context::default());
    resources.insert(PerformanceInfo::new());
    resources.insert(gpu);
    resources.insert(gpu_world);
    resources.insert(SpriteData {
        registry: sprite_registry,
    });
    resources.insert(VisitedChunks(HashSet::new()));
    resources.insert(LoadedAsteroids(HashSet::new()));
    resources.insert(WorldSeed(WORLD_SEED));

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
        .add_system(presence_position_update_system())
        // Flush command buffer so newly-spawned asteroid entities are visible
        // to the render system in the same frame. Without this, legion defers
        // `commands.push(...)` until after the thread-local render, causing
        // freshly-generated asteroids to flash a degenerate quad at the origin.
        .flush()
        .add_thread_local(render_system())
        .build()
}

fn fov_y(w: u32, h: u32) -> f32 {
    2.0 * f32::atan(h as f32 / w as f32)
}

fn main() {
    //env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let (window, mut event_pump) = initialize().unwrap();

    let handle = Arc::new(SdlWindowHandle {
        window: window.window_handle().unwrap().as_raw(),
        display: window.display_handle().unwrap().as_raw(),
    });

    let mut schedule = build_schedule();
    let mut world = World::default();
    let mut resources = initial_resources(handle);
    let _window = window;

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
                    let new_w = x.max(1) as u32;
                    let new_h = y.max(1) as u32;
                    {
                        let mut ss = resources.get_mut::<ScreenState>().unwrap();
                        ss.width = new_w;
                        ss.height = new_h;
                        ss.fov_y_rad = fov_y(new_w, new_h);
                    }
                    resources
                        .get_mut::<GpuRenderer>()
                        .unwrap()
                        .resize(new_w, new_h);
                }
                _ => {}
            }
        }

        schedule.execute(&mut world, &mut resources);
    }
}
