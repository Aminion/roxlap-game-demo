mod components;
mod generation;
mod input;
mod math;
mod sprites;
mod systems;
#[cfg(test)]
mod test_utils;
mod world;

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, IVec3, Vec2};
use legion::{Resources, Schedule, World, *};
use raw_window_handle::{
    DisplayHandle, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    WindowHandle,
};
use roxlap_gpu::SpriteModelRegistry;
use roxlap_render::{GpuRendererSettings, RenderOptions, SceneRenderer, SpriteSet};
use sdl2::{
    event::{Event, WindowEvent},
    keyboard::Scancode,
    mouse::MouseButton,
    video::Window,
    EventPump,
};

use crate::components::{cannon::Cannon, miner::Miner};
use crate::input::PlayerInput;
use crate::systems::{
    aabb::aabb_update_system,
    autopilot::autopilot_system,
    camera::camera_update_system,
    crystal::crystal_system,
    energy::{Energy, ENERGY_MAX},
    lighting::{lighting_system, PointLights},
    miner_input::miner_input_system,
    newton_body::newton_body_system,
    particle::particle_system,
    performance_info::{update_info_system, PerformanceInfo},
    presence_position::presence_position_update_system,
    projectile::projectile_system,
    render::render_system,
    retrieval::retrieval_system,
    shooting::shooting_system,
    thruster::thruster_system,
    ui::ui_system,
};
use crate::world::{
    generate_star_sky, miner_initial_forward, populate_world, register_miner_model,
    register_shared_sprites, CrystalModel, MinerModel, ParticleModel, ProjectileModel,
};

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

/// True while the right mouse button is held — activates the crystal retrieval beam.
pub struct Retrieving(pub bool);

/// Accumulated mouse motion for the current frame, reset before each event poll.
pub type MouseDelta = Vec2;

pub struct FrameTimer(pub Instant);

pub enum GameState {
    TitleScreen,
    Playing,
    GameOver,
}

pub enum CameraMode {
    FirstPerson,
    ThirdPerson,
}

// --- GPU resources ---

/// Set of chunk coordinates (in chunk-space) that have already been visited and populated.
pub struct VisitedChunks(pub HashSet<IVec3>);

/// Set of asteroid entity IDs currently loaded within the presence area.
pub struct LoadedAsteroids(pub HashSet<Entity>);

/// Seed for all procedural world generation (chunk density noise, asteroid properties).
pub struct WorldSeed(pub u64);

/// Tombstoned sprite models accumulated since the last `compact_sprite_models` call.
/// Compact fires when the chunk generation queue empties (or on threshold revisits
/// with pending tombstones), so its cost lands on a frame already paying generation.
pub struct PendingCompact(pub u32);

/// FIFO of chunk coordinates waiting to be generated, drained a few per frame.
pub struct ChunkQueue(pub VecDeque<IVec3>);

/// Set mirror of `ChunkQueue` for O(1) membership tests.
/// Invariant: exactly the chunks currently in `ChunkQueue.0`.
pub struct QueuedChunks(pub HashSet<IVec3>);

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

    let mut renderer = SceneRenderer::new(
        handle,
        (INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT),
        &RenderOptions {
            want_gpu: true,
            gpu: GpuRendererSettings {
                uncapped_present: false,
                ..GpuRendererSettings::default()
            },
            ..RenderOptions::default()
        },
    );
    let (sky_pixels, sky_w, sky_h) = generate_star_sky(WORLD_SEED);
    renderer.set_sky_panorama(&sky_pixels, sky_w, sky_h);

    let world_scene = roxlap_scene::Scene::new();

    let mut sprite_registry = SpriteModelRegistry::new();
    let (projectile_model, crystal_model, particle_model) =
        register_shared_sprites(&mut renderer, &mut sprite_registry);
    let miner_model = register_miner_model(&mut renderer, &mut sprite_registry);

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
    let egui_ctx = egui::Context::default();
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "monocraft".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/Monocraft.ttc")).into(),
    );
    fonts
        .families
        .get_mut(&egui::FontFamily::Monospace)
        .unwrap()
        .insert(0, "monocraft".to_owned());
    fonts
        .families
        .get_mut(&egui::FontFamily::Proportional)
        .unwrap()
        .insert(0, "monocraft".to_owned());
    egui_ctx.set_fonts(fonts);
    egui_ctx.global_style_mut(|style| {
        for text_style in style.text_styles.values_mut() {
            text_style.size = 16.0;
        }
    });
    resources.insert(egui_ctx);
    resources.insert(PerformanceInfo::new());
    resources.insert(renderer);
    resources.insert(world_scene);
    resources.insert(sprite_registry);
    resources.insert(projectile_model);
    resources.insert(crystal_model);
    resources.insert(particle_model);
    resources.insert(miner_model);
    resources.insert(VisitedChunks(HashSet::new()));
    resources.insert(LoadedAsteroids(HashSet::new()));
    resources.insert(WorldSeed(WORLD_SEED));
    resources.insert(PendingCompact(0));
    resources.insert(ChunkQueue(VecDeque::new()));
    resources.insert(QueuedChunks(HashSet::new()));
    resources.insert(Energy::new(ENERGY_MAX));
    resources.insert(Retrieving(false));
    resources.insert(GameState::TitleScreen);
    resources.insert(CameraMode::ThirdPerson);
    resources.insert(PointLights(Vec::new()));

    resources
}

fn build_schedule() -> Schedule {
    Schedule::builder()
        .add_system(update_info_system())
        .add_system(miner_input_system())
        .add_system(autopilot_system())
        .add_system(thruster_system())
        .add_system(retrieval_system())
        .add_system(newton_body_system())
        .add_system(camera_update_system())
        .add_system(presence_position_update_system())
        // Flush so newly-spawned asteroid entities are visible to subsequent systems.
        .flush()
        .add_system(aabb_update_system())
        .add_system(shooting_system())
        .add_system(projectile_system())
        .add_system(crystal_system())
        .add_system(particle_system())
        // Flush so despawned entities are removed before render.
        // lighting_system runs here so crystal lights reflect post-despawn state.
        .flush()
        .add_system(lighting_system())
        .add_thread_local(render_system())
        .add_thread_local(ui_system())
        .build()
}

fn fov_y(w: u32, h: u32) -> f32 {
    2.0 * f32::atan(h as f32 / w as f32)
}

fn restart_world(world: &mut World, resources: &mut Resources) {
    // Reset the renderer sprite registry so model/instance handles restart from
    // a clean slate, matching the fresh CPU SpriteModelRegistry chain_ids.
    {
        let mut renderer = resources.get_mut::<SceneRenderer>().unwrap();
        let _ = renderer.set_sprites(&SpriteSet {
            models: vec![],
            instances: vec![],
            carve_model: None,
        });
    }

    // Reset CPU sprite registry so chain_ids restart from 0.
    *resources.get_mut::<SpriteModelRegistry>().unwrap() = SpriteModelRegistry::new();

    // Re-register the shared projectile/crystal/particle models on the clean slate.
    let (proj, crystal, particle) = {
        let mut renderer = resources.get_mut::<SceneRenderer>().unwrap();
        let mut registry = resources.get_mut::<SpriteModelRegistry>().unwrap();
        register_shared_sprites(&mut renderer, &mut registry)
    };
    *resources.get_mut::<ProjectileModel>().unwrap() = proj;
    *resources.get_mut::<CrystalModel>().unwrap() = crystal;
    *resources.get_mut::<ParticleModel>().unwrap() = particle;

    let miner_model = {
        let mut renderer = resources.get_mut::<SceneRenderer>().unwrap();
        let mut registry = resources.get_mut::<SpriteModelRegistry>().unwrap();
        register_miner_model(&mut renderer, &mut registry)
    };
    *resources.get_mut::<MinerModel>().unwrap() = miner_model;

    // Rebuild ECS world with a fresh miner.
    *world = World::default();
    {
        let miner_model = resources.get::<MinerModel>().unwrap();
        let mut renderer = resources.get_mut::<SceneRenderer>().unwrap();
        populate_world(world, &mut renderer, &miner_model);
    }

    // Reset all runtime resources.
    resources.get_mut::<Energy>().unwrap().current = ENERGY_MAX;
    resources.get_mut::<VisitedChunks>().unwrap().0.clear();
    resources.get_mut::<LoadedAsteroids>().unwrap().0.clear();
    resources.get_mut::<PendingCompact>().unwrap().0 = 0;
    resources.get_mut::<ChunkQueue>().unwrap().0.clear();
    resources.get_mut::<QueuedChunks>().unwrap().0.clear();
    *resources.get_mut::<AutopilotTarget>().unwrap() = AutopilotTarget(miner_initial_forward());
    resources.get_mut::<Retrieving>().unwrap().0 = false;
    *resources.get_mut::<CameraMode>().unwrap() = CameraMode::ThirdPerson;
    resources.get_mut::<HashSet<PlayerInput>>().unwrap().clear();
    *resources.get_mut::<MouseDelta>().unwrap() = Vec2::ZERO;
    resources.get_mut::<FrameTimer>().unwrap().0 = Instant::now();
}

fn main() {
    let (window, mut event_pump) = initialize().unwrap();

    let handle = Arc::new(SdlWindowHandle {
        window: window.window_handle().unwrap().as_raw(),
        display: window.display_handle().unwrap().as_raw(),
    });

    let mut schedule = build_schedule();
    let mut world = World::default();
    let _window = window;
    let mut resources = initial_resources(handle);

    {
        let miner_model = resources.get::<MinerModel>().unwrap();
        let mut renderer = resources.get_mut::<SceneRenderer>().unwrap();
        populate_world(&mut world, &mut renderer, &miner_model);
    }

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
            let playing = matches!(*resources.get::<GameState>().unwrap(), GameState::Playing);
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    scancode: Some(Scancode::Escape),
                    ..
                } => break 'running,
                Event::KeyDown {
                    scancode: Some(Scancode::Return),
                    ..
                } => {
                    let is_title = matches!(
                        *resources.get::<GameState>().unwrap(),
                        GameState::TitleScreen
                    );
                    let is_game_over =
                        matches!(*resources.get::<GameState>().unwrap(), GameState::GameOver);
                    if is_title {
                        *resources.get_mut::<GameState>().unwrap() = GameState::Playing;
                    } else if is_game_over {
                        restart_world(&mut world, &mut resources);
                        *resources.get_mut::<GameState>().unwrap() = GameState::Playing;
                    }
                }
                Event::KeyDown {
                    scancode: Some(Scancode::C),
                    ..
                } if playing => {
                    let mut mode = resources.get_mut::<CameraMode>().unwrap();
                    *mode = match *mode {
                        CameraMode::FirstPerson => CameraMode::ThirdPerson,
                        CameraMode::ThirdPerson => CameraMode::FirstPerson,
                    };
                }
                Event::KeyDown {
                    scancode: Some(code),
                    ..
                } if playing => {
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
                Event::MouseButtonDown {
                    mouse_btn: MouseButton::Left,
                    ..
                } if playing => {
                    let mut q = <(&Miner, &mut Cannon)>::query();
                    for (_, canon) in q.iter_mut(&mut world) {
                        canon.firing = true;
                    }
                }
                Event::MouseButtonUp {
                    mouse_btn: MouseButton::Left,
                    ..
                } => {
                    let mut q = <(&Miner, &mut Cannon)>::query();
                    for (_, canon) in q.iter_mut(&mut world) {
                        canon.firing = false;
                    }
                }
                Event::MouseButtonDown {
                    mouse_btn: MouseButton::Right,
                    ..
                } if playing => {
                    resources.get_mut::<Retrieving>().unwrap().0 = true;
                }
                Event::MouseButtonUp {
                    mouse_btn: MouseButton::Right,
                    ..
                } => {
                    resources.get_mut::<Retrieving>().unwrap().0 = false;
                }
                Event::MouseMotion { xrel, yrel, .. } if playing => {
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
                        .get_mut::<SceneRenderer>()
                        .unwrap()
                        .resize(new_w, new_h);
                }
                _ => {}
            }
        }

        schedule.execute(&mut world, &mut resources);

        {
            let is_playing = matches!(*resources.get::<GameState>().unwrap(), GameState::Playing);
            if is_playing && resources.get::<Energy>().unwrap().current <= 0.0 {
                *resources.get_mut::<GameState>().unwrap() = GameState::GameOver;
            }
        }
    }
}
