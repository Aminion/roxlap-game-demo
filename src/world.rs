use std::f32::consts::{PI, TAU};

use glam::{DMat3, DQuat, DVec3};
use legion::World;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use roxlap_gpu::{camera::Camera as GpuCamera, SpriteModel};

use crate::components::{
    camera::CameraComponent, miner::Miner, newton_body::NewtonBody,
    presence_position::PresencePosition, thruster::ThrusterBank,
};

pub const CUBE_VXL_VSID: u32 = 16;

fn random_voxel_colour(rng: &mut impl rand::Rng) -> u32 {
    let v = rng.random_range(50u32..=180);
    0x80_00_00_00 | (v << 16) | (v << 8) | v
}

/// Generate a 1024×512 equirectangular star panorama as RGBA bytes.
///
/// Stars are placed with uniform solid-angle distribution so the sky
/// looks even from any direction (avoids pole crowding that a naive
/// random-pixel scatter would produce).
pub fn generate_star_sky(seed: u64) -> (Vec<u8>, u32, u32) {
    // u=elevation covers [0,π], v=azimuth covers [0,2π]; set H=2W so
    // both axes have equal angular resolution and stars appear round.
    const W: u32 = 512;
    const H: u32 = 1024;
    const STAR_COUNT: u32 = 1024;

    let mut pixels = vec![0u8; (W * H * 4) as usize];
    let mut rng = StdRng::seed_from_u64(seed);

    for _ in 0..STAR_COUNT {
        // Uniform solid-angle distribution: cos(polar) uniform in [-1, 1].
        let cos_theta: f32 = rng.random_range(-1.0f32..=1.0);
        let phi: f32 = rng.random_range(0.0f32..TAU);

        // Match the UV convention in scene_dda.wgsl:
        //   u = elevation = acos(-dir.z) / π  → texture column (x)
        //   v = azimuth  = atan2(x,y)/(2π)+.5 → texture row    (y)
        let elevation_uv = (-cos_theta).acos() / PI;
        let azimuth_uv = phi / TAU;

        let cx = (elevation_uv * W as f32).min(W as f32 - 1.0) as i32;
        let cy = (azimuth_uv * H as f32) as i32 % H as i32;

        let brightness: u8 = rng.random_range(160u8..=220);
        let b = brightness as f32;

        // Color: 20% red-biased, 20% blue-biased, 60% white.
        let (r, g, bl) = match rng.random_range(0u8..10) {
            0..=1 => ((b * 1.0) as u8, (b * 0.80) as u8, (b * 0.75) as u8),
            2..=3 => ((b * 0.75) as u8, (b * 0.85) as u8, (b * 1.0) as u8),
            _ => (brightness, brightness, brightness),
        };

        // Size: 2, 3, or 4 pixels — larger than before to survive bilinear blur.
        let size: i32 = match rng.random_range(0u8..10) {
            0..=5 => 2,
            6..=8 => 3,
            _ => 4,
        };

        // In equirectangular, azimuth pixels compress by sin(elevation) near the poles.
        // Stretch the star in the azimuth (row) direction by 1/sin(elevation) so it
        // appears round on screen regardless of where on the sphere it sits.
        let sin_elev = (elevation_uv * PI).sin();
        let width_v = ((size as f32 / sin_elev).round() as i32).max(size);
        let half_u = size / 2;
        let half_v = width_v / 2;

        for dx in 0..size {
            // elevation (column) direction
            for dy in 0..width_v {
                // azimuth (row) direction — stretched
                let spx = (cx + dx - half_u).clamp(0, W as i32 - 1) as u32;
                let spy = ((cy + dy - half_v).rem_euclid(H as i32)) as u32;
                let i = ((spy * W + spx) * 4) as usize;
                pixels[i] = r;
                pixels[i + 1] = g;
                pixels[i + 2] = bl;
                pixels[i + 3] = 255;
            }
        }
    }

    (pixels, W, H)
}

pub fn build_asteroid_sprite_model() -> SpriteModel {
    let vsid = CUBE_VXL_VSID as usize;
    let center = CUBE_VXL_VSID as f64 / 2.0;
    let radius = center - 0.5;

    let mx = CUBE_VXL_VSID;
    let my = CUBE_VXL_VSID;
    let mz = CUBE_VXL_VSID;
    let occ_words_per_col = mz.div_ceil(32).max(1);
    let cols = (mx * my) as usize;

    let mut occupancy = vec![0u32; cols * occ_words_per_col as usize];
    let mut color_offsets = vec![0u32; cols + 1];
    let mut colors: Vec<u32> = Vec::new();
    let mut dirs: Vec<u32> = Vec::new();

    let mut rng = rand::rng();
    for y in 0..vsid {
        for x in 0..vsid {
            let col = x + y * vsid;
            color_offsets[col] = colors.len() as u32;
            for z in 0..vsid {
                let dx = x as f64 + 0.5 - center;
                let dy = y as f64 + 0.5 - center;
                let dz = z as f64 + 0.5 - center;
                if dx * dx + dy * dy + dz * dz <= radius * radius {
                    occupancy[col * occ_words_per_col as usize + z / 32] |= 1u32 << (z % 32);
                    colors.push(random_voxel_colour(&mut rng));
                    dirs.push(0);
                }
            }
        }
    }
    color_offsets[cols] = colors.len() as u32;

    SpriteModel {
        dims: [mx, my, mz],
        occ_words_per_col,
        pivot: [center as f32, center as f32, center as f32],
        occupancy,
        colors,
        dirs,
        color_offsets,
        voxel_world_size: 1.0,
    }
}

pub fn populate_world(world: &mut World) {
    spawn_miner(world);
}

const MINER_PITCH: f64 = 0.8;
const MINER_SPAWN_OFFSET_X: f64 = 70.0;
const MINER_SPAWN_HEIGHT: f64 = 100.0;

fn miner_orientation() -> DQuat {
    let (sp, cp) = (MINER_PITCH.sin(), MINER_PITCH.cos());
    DQuat::from_mat3(&DMat3::from_cols(
        DVec3::Y,
        DVec3::new(-sp, 0.0, cp),
        DVec3::new(-cp, 0.0, -sp),
    ))
    .normalize()
}

pub fn miner_initial_forward() -> DVec3 {
    miner_orientation() * DVec3::NEG_Z
}

fn spawn_miner(world: &mut World) {
    let orientation = miner_orientation();
    let pos = DVec3::new(-MINER_SPAWN_OFFSET_X, 0.0, -MINER_SPAWN_HEIGHT);
    // CameraComponent is overwritten by camera_update_system before the first render,
    // so the initial values are placeholders.
    world.push((
        Miner,
        NewtonBody {
            mass: 1.0,
            pos,
            vel: DVec3::ZERO,
            orientation,
            angular_vel: DVec3::ZERO,
        },
        CameraComponent(GpuCamera {
            position: [0.0; 3],
            forward: [0.0, 0.0, -1.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            fov_y_rad: 0.0,
        }),
        // mass=1.0 kg, radius=1.0 m, rot=0.6 N → 3.0 rad/s² max; lin=5.0 N → 5.0 m/s² max
        ThrusterBank::new(1.0, 1.0, 0.6, 5.0),
        // Infinity forces the distance check to fire on frame 1, generating the initial chunks.
        PresencePosition(DVec3::splat(f64::INFINITY)),
    ));
}
