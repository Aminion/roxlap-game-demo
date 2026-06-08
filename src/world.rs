use glam::{DMat3, DQuat, DVec3};
use legion::World;
use rand::RngExt;
use roxlap_cavegen::pack_dense_grid_to_vxl;
use roxlap_core::Camera;
use roxlap_formats::{edit::MAXZDIM, vxl::Vxl};

use crate::components::{
    camera::CameraComponent, cube_marker::CubeMarker, miner::Miner, newton_body::NewtonBody,
    thruster::ThrusterBank,
};

pub const VSID: u32 = 64;

/// Z-coord of the (one-voxel-thick) ground plane. Voxlap is **z-down**:
/// small z is up, large z is down.
pub const GROUND_Z: i32 = 200;

pub const CUBE_VXL_VSID: u32 = 16;
pub const CUBE_VXL_EDGE: i32 = 16;

/// Holds both the static ground world and the pre-lit cube VXL.
/// Bundled into one resource to stay within Legion's 8-resource-per-system limit.
pub struct Worlds {
    pub base: Vxl,
    pub cube: Vxl,
}

fn random_voxel_colour(rng: &mut impl rand::Rng) -> u32 {
    0x80_00_00_00 | (rng.random::<u32>() & 0x00_FF_FF_FF)
}

fn voxel_idx(x: usize, y: usize, z: usize, vsid: usize) -> usize {
    (y * vsid + x) * MAXZDIM as usize + z
}

pub fn build_world() -> Vxl {
    let vsid_u = VSID as usize;
    let cells = vsid_u * vsid_u * MAXZDIM as usize;

    let mut mask = vec![0u8; cells];
    let mut colour = vec![0u32; cells];
    let mut rng = rand::rng();
    for y in 0..vsid_u {
        for x in 0..vsid_u {
            let i = voxel_idx(x, y, GROUND_Z as usize, vsid_u);
            mask[i] = 1;
            colour[i] = random_voxel_colour(&mut rng);
        }
    }
    pack_dense_grid_to_vxl(&mask, &colour, VSID)
}

pub fn build_cube_vxl() -> Vxl {
    let vsid = CUBE_VXL_VSID as usize;
    let cells = vsid * vsid * MAXZDIM as usize;

    let mut mask = vec![0u8; cells];
    let mut colour = vec![0u32; cells];

    let center = CUBE_VXL_VSID as f64 / 2.0;
    let radius = center - 0.5;

    let mut rng = rand::rng();
    for y in 0..vsid {
        for x in 0..vsid {
            for z in 0..vsid {
                let dx = x as f64 + 0.5 - center;
                let dy = y as f64 + 0.5 - center;
                let dz = z as f64 + 0.5 - center;
                if dx * dx + dy * dy + dz * dz <= radius * radius {
                    let i = voxel_idx(x, y, z, vsid);
                    mask[i] = 1;
                    colour[i] = random_voxel_colour(&mut rng);
                }
            }
        }
    }
    pack_dense_grid_to_vxl(&mask, &colour, CUBE_VXL_VSID)
}

pub fn populate_world(world: &mut World) {
    spawn_miner(world);
    spawn_cube(world);
}

const MINER_PITCH: f64 = 0.8;
/// Lateral offset from world-center so the miner spawns clear of the cube.
const MINER_SPAWN_OFFSET_X: f64 = 70.0;
/// Height above the ground plane at which the miner spawns (world units).
const MINER_SPAWN_HEIGHT: f64 = 100.0;
/// Vertical clearance between the ground and the bottom of the cube on spawn (world units).
const CUBE_SPAWN_CLEARANCE: f64 = 15.0;

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
    let pos = DVec3::new(
        f64::from(VSID) * 0.5 - MINER_SPAWN_OFFSET_X,
        f64::from(VSID) * 0.5,
        f64::from(GROUND_Z) - MINER_SPAWN_HEIGHT,
    );
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
        CameraComponent(Camera {
            pos: [0.0; 3],
            forward: [0.0, 0.0, -1.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
        }),
        // rot: radius=1.0 m, 0.6 N × 2 nozzles → 3.0 rad/s² max
        // lin: 5.0 N → 5.0 m/s² max linear acceleration
        ThrusterBank::new(1.0, 0.6, 5.0),
    ));
}

fn spawn_cube(world: &mut World) {
    let pos = DVec3::new(
        f64::from(VSID) / 2.0,
        f64::from(VSID) / 2.0,
        f64::from(GROUND_Z) - f64::from(CUBE_VXL_EDGE) / 2.0 - CUBE_SPAWN_CLEARANCE,
    );
    world.push((
        CubeMarker,
        NewtonBody {
            mass: 1.0,
            pos,
            vel: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_vel: DVec3::new(0.3, 0.2, 0.1),
        },
    ));
}
