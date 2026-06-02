use legion::{system, world::SubWorld};
use roxlap_core::{
    opticast, rasterizer::ScratchPool, scalar_rasterizer::ScalarRasterizer, update_lighting,
    Camera, Engine, GridView, OpticastSettings,
};
use roxlap_formats::{edit::MAXZDIM, vxl::Vxl};
use sdl2::pixels::PixelFormatEnum;

use crate::{components::miner::Miner, CanvasResources};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// World footprint in voxels along X and Y. Voxlap requires a power
/// of two; 32 keeps the bake fast and leaves room around the cube.
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

struct Cam {
    pos: [f64; 3],
    yaw: f64,
    pitch: f64,
}

impl Cam {
    /// Compose voxlap's right-handed yaw / pitch basis:
    ///
    /// - `right × down = forward` (chirality the engine's frustum
    ///   math assumes — flip and sprites + side shades silently
    ///   render upside-down).
    /// - `yaw = 0` looks `+x`; positive `yaw` rotates toward `+y`.
    /// - `pitch = 0` is level; positive `pitch` tilts the view down.
    fn camera(&self) -> Camera {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = [cy * cp, sy * cp, sp];
        let right = [-sy, cy, 0.0];
        let down = [
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        ];
        Camera {
            pos: self.pos,
            right,
            down,
            forward,
        }
    }
}

#[system]
#[read_component(Miner)]
pub fn render(
    #[resource] canvas_resources: &mut CanvasResources,
    #[resource] world_map: &mut Vxl,
    #[resource] engine: &mut Engine,
    world: &SubWorld,
) {
    update_lighting(
        &mut world_map.data,
        &world_map.column_offset,
        world_map.vsid,
        0,
        0,
        0,
        world_map.vsid as i32,
        world_map.vsid as i32,
        MAXZDIM,
        engine.lightmode(),
        engine.lights(),
    );

    let cx = f64::from(VSID) * 0.5;
    let cy = f64::from(VSID) * 0.5;
    let cz = f64::from(GROUND_Z) - f64::from(CUBE_EDGE) - 6.0;
    let cam = Cam {
        pos: [cx - 16.0, cy, cz],
        yaw: 0.0,
        pitch: 0.15, // tilt down a touch to put the cube in frame
    };

    let pool = ScratchPool::new(WIDTH, HEIGHT, world_map.vsid);
    // 4. Setup Camera Vectors (Position, Forward, Right, Down)
    // Voxlap looks for directional vectors to calculate view matrices

    let mut texture = canvas_resources
        .texture_creator
        .create_texture_streaming(PixelFormatEnum::ARGB8888, WIDTH, HEIGHT)
        .map_err(|e| e.to_string())
        .unwrap();
    let mut pool = ScratchPool::new(WIDTH, HEIGHT, world_map.vsid);
    let mut framebuffer: Vec<u32> = vec![0u32; (WIDTH * HEIGHT) as usize];
    let mut zbuffer: Vec<f32> = vec![0.0f32; (WIDTH * HEIGHT) as usize];
    let camera = cam.camera();
    let settings = OpticastSettings::for_oracle_framebuffer(WIDTH, HEIGHT);

    {
        let grid = GridView::from_single_vxl(&world_map);
        let mut rasterizer =
            ScalarRasterizer::new(&mut framebuffer, &mut zbuffer, WIDTH as usize, grid);
        let _ = opticast(&mut rasterizer, &mut pool, &camera, &settings, grid);
    }

    let pitch = (WIDTH * 4) as usize; // 4 bytes per pixel (ARGB8888)
    let u8_bytes: &[u8] = bytemuck::cast_slice(&framebuffer);
    texture
        .update(None, u8_bytes, pitch)
        .expect("Failed to update texture");

    // 7. Flush the streaming texture onto screen VRAM
    canvas_resources.canvas.clear();
    canvas_resources.canvas.copy(&texture, None, None).unwrap();
    canvas_resources.canvas.present();
}
