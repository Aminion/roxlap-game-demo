use std::f32::consts::{PI, TAU};

use glam::{DMat3, DQuat, DVec3, UVec3};
use legion::World;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use roxlap_cavegen::PerlinNoise3D;
use roxlap_gpu::{camera::Camera as GpuCamera, SpriteModel};

use crate::components::{
    camera::CameraComponent, canon::Canon, miner::Miner, newton_body::NewtonBody,
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

/// One red voxel is scattered per this many occupied voxels per mineral point,
/// giving a density of `mineral_count / RED_VOXELS_PER_MINERAL` across the sphere.
const RED_VOXELS_PER_MINERAL: f32 = 20.0;

/// Perlin noise frequency for asteroid surface distortion, in voxel-space.
/// Wavelength ≈ 1 / 0.20 ≈ 5 voxels — about 3 bumps across the 16-voxel diameter.
const ASTEROID_NOISE_FREQ: f32 = 0.20;

/// fBm octaves for asteroid surface noise.
const ASTEROID_NOISE_OCTAVES: u32 = 3;

/// Max surface displacement in voxels. fBm peaks near ±0.866, so effective
/// range is ≈ ±3.0 voxels against a base radius of 7.5.
const ASTEROID_NOISE_AMP: f64 = 3.5;

/// Minimum voxel distance inside the displaced surface required for a mineral point.
const MINERAL_SURFACE_BUFFER: f64 = 2.0;

pub fn build_asteroid_sprite_model(
    seed: u64,
    noise_seed: u64,
    scale_seed: u64,
    mineral_count: usize,
) -> SpriteModel {
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

    let red_prob = mineral_count as f32 / RED_VOXELS_PER_MINERAL;
    let mut rng = StdRng::seed_from_u64(seed);
    let perlin = PerlinNoise3D::new(noise_seed);
    let (sx, sy, sz) = {
        let mut srng = StdRng::seed_from_u64(scale_seed);
        (
            srng.random_range(0.7f64..=1.3),
            srng.random_range(0.7f64..=1.3),
            srng.random_range(0.7f64..=1.3),
        )
    };
    for y in 0..vsid {
        for x in 0..vsid {
            let col = x + y * vsid;
            color_offsets[col] = colors.len() as u32;
            for z in 0..vsid {
                let dx = x as f64 + 0.5 - center;
                let dy = y as f64 + 0.5 - center;
                let dz = z as f64 + 0.5 - center;
                let d =
                    ((dx / sx) * (dx / sx) + (dy / sy) * (dy / sy) + (dz / sz) * (dz / sz)).sqrt();
                let noise = perlin.fbm(
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                    z as f32 + 0.5,
                    ASTEROID_NOISE_OCTAVES,
                    ASTEROID_NOISE_FREQ,
                );
                let noisy_r = radius + noise as f64 * ASTEROID_NOISE_AMP;
                if d <= noisy_r {
                    occupancy[col * occ_words_per_col as usize + z / 32] |= 1u32 << (z % 32);
                    let color = if red_prob > 0.0 && rng.random::<f32>() < red_prob {
                        0x80_C0_30_30
                    } else {
                        random_voxel_colour(&mut rng)
                    };
                    colors.push(color);
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

/// Return 3–5 model-local voxel positions buried inside the asteroid.
/// Uses the same Perlin surface displacement as `build_asteroid_sprite_model`
/// and requires a `MINERAL_SURFACE_BUFFER` voxel margin inside the actual surface.
pub fn generate_mineral_points(
    vsid: u32,
    seed: u64,
    noise_seed: u64,
    scale_seed: u64,
) -> Vec<UVec3> {
    let mut rng = StdRng::seed_from_u64(seed);
    let center = vsid as f64 / 2.0;
    let radius = center - 0.5;
    let perlin = PerlinNoise3D::new(noise_seed);
    let (sx, sy, sz) = {
        let mut srng = StdRng::seed_from_u64(scale_seed);
        (
            srng.random_range(0.7f64..=1.3),
            srng.random_range(0.7f64..=1.3),
            srng.random_range(0.7f64..=1.3),
        )
    };

    let mut candidates: Vec<UVec3> = Vec::new();
    for y in 0..vsid {
        for x in 0..vsid {
            for z in 0..vsid {
                let dx = x as f64 + 0.5 - center;
                let dy = y as f64 + 0.5 - center;
                let dz = z as f64 + 0.5 - center;
                let d =
                    ((dx / sx) * (dx / sx) + (dy / sy) * (dy / sy) + (dz / sz) * (dz / sz)).sqrt();
                let noise = perlin.fbm(
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                    z as f32 + 0.5,
                    ASTEROID_NOISE_OCTAVES,
                    ASTEROID_NOISE_FREQ,
                );
                let noisy_r = radius + noise as f64 * ASTEROID_NOISE_AMP;
                if d < noisy_r - MINERAL_SURFACE_BUFFER {
                    candidates.push(UVec3::new(x, y, z));
                }
            }
        }
    }

    let count = (rng.random_range(3u32..=5) as usize).min(candidates.len());
    for i in 0..count {
        let j = rng.random_range(i..candidates.len());
        candidates.swap(i, j);
    }
    candidates.truncate(count);
    candidates
}

/// 7-voxel cross crystal: one centre voxel plus one on each of the six faces.
pub fn build_crystal_sprite_model() -> SpriteModel {
    const DIM: u32 = 3;
    let occ_words_per_col = DIM.div_ceil(32).max(1);
    let cols = (DIM * DIM) as usize;

    let mut occupancy = vec![0u32; cols * occ_words_per_col as usize];
    let mut color_offsets = vec![0u32; cols + 1];
    let mut colors: Vec<u32> = Vec::new();
    let mut dirs: Vec<u32> = Vec::new();

    // (x, y, z) for the 7 arm voxels, sorted by column then ascending z.
    let arm_voxels: &[(u32, u32, u32)] = &[
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 1),
        (2, 1, 1),
    ];

    for y in 0..DIM {
        for x in 0..DIM {
            let col = (x + y * DIM) as usize;
            color_offsets[col] = colors.len() as u32;
            let mut zs: Vec<u32> = arm_voxels
                .iter()
                .filter(|&&(vx, vy, _)| vx == x && vy == y)
                .map(|&(_, _, vz)| vz)
                .collect();
            zs.sort_unstable();
            for vz in zs {
                let base = col * occ_words_per_col as usize + (vz / 32) as usize;
                occupancy[base] |= 1u32 << (vz % 32);
                colors.push(0x80_FF_30_30); // red crystal
                dirs.push(0);
            }
        }
    }
    color_offsets[cols] = colors.len() as u32;

    SpriteModel {
        dims: [DIM, DIM, DIM],
        occ_words_per_col,
        pivot: [1.5, 1.5, 1.5],
        occupancy,
        colors,
        dirs,
        color_offsets,
        voxel_world_size: 1.0,
    }
}

pub fn build_projectile_sprite_model() -> SpriteModel {
    SpriteModel {
        dims: [1, 1, 1],
        occ_words_per_col: 1,
        pivot: [0.5, 0.5, 0.5],
        occupancy: vec![1u32],
        color_offsets: vec![0u32, 1u32],
        colors: vec![0x80_FF_00_FF],
        dirs: vec![0u32],
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
        Canon {
            firing: false,
            cooldown: 0.0,
        },
    ));
}
