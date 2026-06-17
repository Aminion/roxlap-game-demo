use glam::{DMat3, DQuat, DVec3};
use legion::World;
use rand::RngExt;
use roxlap_gpu::{camera::Camera as GpuCamera, SpriteModel};

use crate::components::{
    camera::CameraComponent, miner::Miner, newton_body::NewtonBody,
    presence_position::PresencePosition, thruster::ThrusterBank,
};

pub const CUBE_VXL_VSID: u32 = 16;

fn random_voxel_colour(rng: &mut impl rand::Rng) -> u32 {
    0x80_00_00_00 | (rng.random::<u32>() & 0x00_FF_FF_FF)
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
