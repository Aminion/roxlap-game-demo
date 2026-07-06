use glam::{DVec3, UVec3};
use rand::{rngs::StdRng, seq::SliceRandom, RngExt, SeedableRng};
use roxlap_cavegen::PerlinNoise3D;
use roxlap_gpu::SpriteModel;

// ── Asteroid ─────────────────────────────────────────────────────────────────

pub const ASTEROID_VOXEL_SIZE: u32 = 16;

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

fn random_voxel_colour(rng: &mut impl rand::Rng) -> u32 {
    let v = rng.random_range(50u32..=180);
    0x80_00_00_00 | (v << 16) | (v << 8) | v
}

fn asteroid_scale(scale_seed: u64) -> DVec3 {
    let mut srng = StdRng::seed_from_u64(scale_seed);
    DVec3::new(
        srng.random_range(0.7f64..=1.3),
        srng.random_range(0.7f64..=1.3),
        srng.random_range(0.7f64..=1.3),
    )
}

/// Returns `noisy_r − d` for voxel `(x, y, z)`. Positive = inside the surface.
fn asteroid_surface_depth(
    x: u32,
    y: u32,
    z: u32,
    center: f64,
    radius: f64,
    scale: DVec3,
    perlin: &PerlinNoise3D,
) -> f64 {
    let d = ((UVec3::new(x, y, z).as_dvec3() + DVec3::splat(0.5) - DVec3::splat(center)) / scale)
        .length();
    let noise = perlin.fbm(
        x as f32 + 0.5,
        y as f32 + 0.5,
        z as f32 + 0.5,
        ASTEROID_NOISE_OCTAVES,
        ASTEROID_NOISE_FREQ,
    );
    radius + noise as f64 * ASTEROID_NOISE_AMP - d
}

/// Build the asteroid sprite model and (when `collect_minerals` is true) select
/// 3–5 embedded mineral positions in a single Perlin pass.
///
/// `color_seed` drives voxel colour RNG; `mineral_seed` drives the mineral
/// count and shuffle. Both must match what `compute_chunk` passes so that
/// colours and mineral positions remain deterministic per chunk hash.
pub fn build_asteroid(
    color_seed: u64,
    mineral_seed: u64,
    noise_seed: u64,
    scale_seed: u64,
    collect_minerals: bool,
) -> (SpriteModel, Vec<UVec3>) {
    let vsid = ASTEROID_VOXEL_SIZE;
    let center = vsid as f64 / 2.0;
    let radius = center - 0.5;
    let occ_words_per_col = vsid.div_ceil(32).max(1);
    let cols = (vsid * vsid) as usize;

    let perlin = PerlinNoise3D::new(noise_seed);
    let scale = asteroid_scale(scale_seed);

    // Pass 1: single Perlin scan — occupancy, per-column voxel counts, mineral candidates.
    let mut occupancy = vec![0u32; cols * occ_words_per_col as usize];
    let mut col_voxel_counts: Vec<u32> = Vec::with_capacity(cols);
    let mut mineral_candidates: Vec<UVec3> =
        Vec::with_capacity(if collect_minerals { 64 } else { 0 });

    for y in 0..vsid {
        for x in 0..vsid {
            let col = (x + y * vsid) as usize;
            let mut count: u32 = 0;
            for z in 0..vsid {
                let depth = asteroid_surface_depth(x, y, z, center, radius, scale, &perlin);
                if depth >= 0.0 {
                    occupancy[col * occ_words_per_col as usize + z as usize / 32] |=
                        1u32 << (z % 32);
                    count += 1;
                    if collect_minerals && depth > MINERAL_SURFACE_BUFFER {
                        mineral_candidates.push(UVec3::new(x, y, z));
                    }
                }
            }
            col_voxel_counts.push(count);
        }
    }

    // Pick minerals now that candidates are known.
    let mut mineral_rng = StdRng::seed_from_u64(mineral_seed);
    if collect_minerals && !mineral_candidates.is_empty() {
        let count = (mineral_rng.random_range(3u32..=5) as usize).min(mineral_candidates.len());
        let _ = mineral_candidates.partial_shuffle(&mut mineral_rng, count);
        mineral_candidates.truncate(count);
    } else {
        mineral_candidates.clear();
    }

    // Pass 2: colour assignment — cheap, no Perlin; pre-allocated from exact voxel count.
    let total_voxels: usize = col_voxel_counts.iter().map(|&c| c as usize).sum();
    let red_prob = mineral_candidates.len() as f32 / RED_VOXELS_PER_MINERAL;
    let mut rng = StdRng::seed_from_u64(color_seed);
    let mut color_offsets = vec![0u32; cols + 1];
    let mut colors: Vec<u32> = Vec::with_capacity(total_voxels);
    let dirs: Vec<u32> = vec![0u32; total_voxels];

    for (col, &count) in col_voxel_counts.iter().enumerate() {
        color_offsets[col] = colors.len() as u32;
        for _ in 0..count {
            let color = if red_prob > 0.0 && rng.random::<f32>() < red_prob {
                0x80_C0_30_30
            } else {
                random_voxel_colour(&mut rng)
            };
            colors.push(color);
        }
    }
    color_offsets[cols] = colors.len() as u32;

    let model = SpriteModel {
        dims: [vsid, vsid, vsid],
        occ_words_per_col,
        pivot: [center as f32; 3],
        occupancy,
        colors,
        dirs,
        color_offsets,
        materials: Vec::new(),
        voxel_world_size: 1.0,
    };

    (model, mineral_candidates)
}

// ── Crystal ───────────────────────────────────────────────────────────────────

/// 7-voxel cross crystal: one centre voxel plus one on each of the six faces.
pub fn build_crystal() -> SpriteModel {
    const DIM: u32 = 3;
    let occ_words_per_col = DIM.div_ceil(32).max(1);
    let cols = (DIM * DIM) as usize;

    let mut occupancy = vec![0u32; cols * occ_words_per_col as usize];
    let mut color_offsets = vec![0u32; cols + 1];
    let mut colors: Vec<u32> = Vec::new();
    let mut dirs: Vec<u32> = Vec::new();

    // (x, y, z) for the 7 arm voxels, sorted by column then ascending z.
    let arm_voxels: &[UVec3] = &[
        UVec3::new(0, 1, 1),
        UVec3::new(1, 0, 1),
        UVec3::new(1, 1, 0),
        UVec3::new(1, 1, 1),
        UVec3::new(1, 1, 2),
        UVec3::new(1, 2, 1),
        UVec3::new(2, 1, 1),
    ];

    for y in 0..DIM {
        for x in 0..DIM {
            let col = (x + y * DIM) as usize;
            color_offsets[col] = colors.len() as u32;
            let mut zs: Vec<u32> = arm_voxels
                .iter()
                .filter(|v| v.x == x && v.y == y)
                .map(|v| v.z)
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
        materials: Vec::new(),
        voxel_world_size: 1.0,
    }
}

// ── Particle ──────────────────────────────────────────────────────────────────

/// 3×3×3 solid cube with a Z-gradient baked in: z=0 (voxlap "up") is bright,
/// z=2 (voxlap "down") is dark. Simulates top-down lighting; tumbling via
/// angular_vel cycles the bright/shadow faces. Each particle instance scales
/// by 1/3 relative to the 1×1×1 case to keep the same world-space size.
pub fn build_particle() -> SpriteModel {
    const DIM: u32 = 3;
    let cols = (DIM * DIM) as usize;
    // Z-gradient: bright top (z=0 = voxlap up), dark bottom (z=2 = voxlap down).
    let z_colors = [0x80_D0_D0_D0u32, 0x80_A0_A0_A0u32, 0x80_60_60_60u32];
    SpriteModel {
        dims: [DIM, DIM, DIM],
        occ_words_per_col: 1,
        pivot: [1.5, 1.5, 1.5],
        // Each column has bits 0,1,2 set (z=0..2 all occupied).
        occupancy: vec![0b111u32; cols],
        // 3 colors per column, 9 columns → 27 entries + 1 sentinel.
        color_offsets: (0..=cols).map(|i| (i * DIM as usize) as u32).collect(),
        colors: (0..cols).flat_map(|_| z_colors).collect(),
        dirs: vec![0u32; cols * DIM as usize],
        materials: Vec::new(),
        voxel_world_size: 1.0,
    }
}

// ── Projectile ────────────────────────────────────────────────────────────────

pub fn build_projectile() -> SpriteModel {
    SpriteModel {
        dims: [1, 1, 1],
        occ_words_per_col: 1,
        pivot: [0.5, 0.5, 0.5],
        occupancy: vec![1u32],
        color_offsets: vec![0u32, 1u32],
        colors: vec![0x80_FF_00_FF],
        dirs: vec![0u32],
        materials: Vec::new(),
        voxel_world_size: 1.0,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn asteroid(collect_minerals: bool) -> (SpriteModel, Vec<UVec3>) {
        build_asteroid(0, 0, 0, 0, collect_minerals)
    }

    // ── structure ─────────────────────────────────────────────────────────────

    #[test]
    fn model_dims_correct() {
        let (m, _) = asteroid(false);
        assert_eq!(m.dims, [ASTEROID_VOXEL_SIZE; 3]);
    }

    #[test]
    fn pivot_at_center() {
        let (m, _) = asteroid(false);
        let expected = ASTEROID_VOXEL_SIZE as f32 / 2.0;
        assert_eq!(m.pivot, [expected; 3]);
    }

    #[test]
    fn model_is_nonempty() {
        let (m, _) = asteroid(false);
        assert!(!m.colors.is_empty());
    }

    #[test]
    fn color_offsets_consistent() {
        let (m, _) = asteroid(true);
        let n_cols = (m.dims[0] * m.dims[1]) as usize;
        assert_eq!(m.color_offsets.len(), n_cols + 1);
        assert_eq!(*m.color_offsets.last().unwrap() as usize, m.colors.len());
    }

    // ── mineral behavior ──────────────────────────────────────────────────────

    #[test]
    fn no_minerals_when_disabled() {
        let (_, minerals) = asteroid(false);
        assert!(minerals.is_empty());
    }

    #[test]
    fn mineral_count_in_range_when_enabled() {
        for seed in [0u64, 1, 42, 0xdead_beef] {
            let (_, minerals) = build_asteroid(seed, seed, seed, seed, true);
            assert!(
                (3..=5).contains(&minerals.len()),
                "expected 3–5 minerals, got {} with seed {seed}",
                minerals.len()
            );
        }
    }

    // ── invariants ────────────────────────────────────────────────────────────

    #[test]
    fn minerals_within_model_bounds() {
        let vsid = ASTEROID_VOXEL_SIZE;
        let (_, minerals) = asteroid(true);
        for &m in &minerals {
            assert!(
                m.cmplt(UVec3::splat(vsid)).all(),
                "mineral {m} is out of model bounds"
            );
        }
    }

    #[test]
    fn minerals_are_occupied_voxels() {
        let (model, minerals) = asteroid(true);
        let vsid = ASTEROID_VOXEL_SIZE;
        let occ_words = model.occ_words_per_col as usize;
        for &m in &minerals {
            let col = (m.x + m.y * vsid) as usize;
            let word = model.occupancy[col * occ_words + m.z as usize / 32];
            let occupied = (word >> (m.z % 32)) & 1 == 1;
            assert!(occupied, "mineral at {m} is not an occupied voxel");
        }
    }

    // ── determinism ───────────────────────────────────────────────────────────

    #[test]
    fn output_is_deterministic() {
        let (m1, min1) = build_asteroid(42, 7, 13, 99, true);
        let (m2, min2) = build_asteroid(42, 7, 13, 99, true);
        assert_eq!(m1.colors, m2.colors);
        assert_eq!(min1, min2);
    }

    // ── build_crystal ─────────────────────────────────────────────────────────

    #[test]
    fn crystal_dims_and_pivot() {
        let m = build_crystal();
        assert_eq!(m.dims, [3, 3, 3]);
        assert_eq!(m.pivot, [1.5, 1.5, 1.5]);
    }

    #[test]
    fn crystal_voxel_count() {
        // 1 centre + 6 face neighbours = 7 voxels.
        assert_eq!(build_crystal().colors.len(), 7);
    }

    #[test]
    fn crystal_color_offsets_consistent() {
        let m = build_crystal();
        let n_cols = (m.dims[0] * m.dims[1]) as usize;
        assert_eq!(m.color_offsets.len(), n_cols + 1);
        assert_eq!(*m.color_offsets.last().unwrap() as usize, m.colors.len());
    }

    #[test]
    fn crystal_arm_voxels_all_occupied() {
        let m = build_crystal();
        let arm: &[UVec3] = &[
            UVec3::new(0, 1, 1),
            UVec3::new(1, 0, 1),
            UVec3::new(1, 1, 0),
            UVec3::new(1, 1, 1),
            UVec3::new(1, 1, 2),
            UVec3::new(1, 2, 1),
            UVec3::new(2, 1, 1),
        ];
        for &v in arm {
            let col = (v.x + v.y * m.dims[0]) as usize;
            let word = m.occupancy[col * m.occ_words_per_col as usize + v.z as usize / 32];
            assert!((word >> (v.z % 32)) & 1 == 1, "voxel {v} not occupied");
        }
    }
}
