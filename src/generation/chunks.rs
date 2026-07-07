use std::collections::{HashSet, VecDeque};

use glam::{DVec3, IVec3, UVec3};
use legion::Entity;
use roxlap_cavegen::PerlinNoise3D;
use roxlap_gpu::SpriteModel;

use crate::{
    math::{hash_to_signed, splitmix64},
    sprites::build_asteroid,
};

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

/// FIFO queue of chunk coordinates waiting to be generated, with an O(1) membership set.
/// Both structures are kept in sync automatically via the `enqueue`/`pop_front`/`drain_front`
/// methods, eliminating the manual invariant previously split across two separate resources.
pub struct ChunkQueue {
    queue: VecDeque<IVec3>,
    queued: HashSet<IVec3>,
}

impl ChunkQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            queued: HashSet::new(),
        }
    }

    /// Push `chunk` unless it is already queued. O(1) duplicate check.
    pub fn enqueue(&mut self, chunk: IVec3) {
        if self.queued.insert(chunk) {
            self.queue.push_back(chunk);
        }
    }

    pub fn pop_front(&mut self) -> Option<IVec3> {
        let chunk = self.queue.pop_front()?;
        self.queued.remove(&chunk);
        Some(chunk)
    }

    pub fn front(&self) -> Option<&IVec3> {
        self.queue.front()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Drain the first `n` entries and remove them from the queued set.
    pub fn drain_front(&mut self, n: usize) -> Vec<IVec3> {
        let chunks: Vec<IVec3> = self.queue.drain(..n).collect();
        for &c in &chunks {
            self.queued.remove(&c);
        }
        chunks
    }

    pub fn clear(&mut self) {
        self.queue.clear();
        self.queued.clear();
    }
}

/// Side length of one chunk in world units.
pub const CHUNK_SIZE: i32 = 64;

/// Radius (in chunks) of the loaded sphere around the player.
pub const LOAD_RADIUS: i32 = 8;

/// Convert a world-space position to the chunk coordinate that contains it.
pub fn world_to_chunk(world_pos: DVec3) -> IVec3 {
    (world_pos / CHUNK_SIZE as f64).floor().as_ivec3()
}

/// Return all chunk coords within `radius` chunks of `center` (inclusive, Euclidean sphere).
pub fn chunks_in_sphere(center: IVec3, radius: i32) -> impl Iterator<Item = IVec3> {
    let r2 = radius * radius;
    (-radius..=radius).flat_map(move |dx| {
        (-radius..=radius).flat_map(move |dy| {
            (-radius..=radius).filter_map(move |dz| {
                let d = IVec3::new(dx, dy, dz);
                (d.dot(d) <= r2).then_some(center + d)
            })
        })
    })
}

/// Return chunk coords within `radius` chunks of `ship_pos` that have not yet been visited.
pub fn missing_chunks<'a>(
    ship_pos: DVec3,
    radius: i32,
    visited: &'a HashSet<IVec3>,
) -> impl Iterator<Item = IVec3> + 'a {
    let center = world_to_chunk(ship_pos);
    chunks_in_sphere(center, radius).filter(move |c| !visited.contains(c))
}

/// Base spatial frequency of the density noise. 1/0.03 ≈ 33 chunks per noise
/// wavelength — large-scale void/dense structure roughly twice the load sphere.
const CHUNK_NOISE_FREQ: f32 = 0.03;

/// fBm octave count. Each octave doubles the frequency, adding finer patchiness:
/// octave 1 ≈ 33-chunk blobs, octave 2 ≈ 16-chunk, octave 3 ≈ 8-chunk.
const CHUNK_NOISE_OCTAVES: u32 = 3;

/// Perlin outputs ≈ ±0.866 (theoretical max √3/2); divide by this to normalise to ±1.
const PERLIN_MAX: f32 = 0.866;

/// Fraction of asteroids that contain crystal deposits. Range [0.0, 1.0].
const CRYSTAL_SPAWN_CHANCE: f32 = 0.01;

/// Max axis offset from chunk centre: CHUNK_SIZE/2 − half_extent = 32 − 8 = 24.
/// Worst-case gap between adjacent asteroids = 64 − 2×24 = 16 = 2×half_extent (touching, not overlapping).
const SPAWN_SAFE_RANGE: f64 = 24.0;

pub(crate) enum ChunkComputeResult {
    NoSpawn {
        chunk: IVec3,
    },
    Spawn {
        chunk: IVec3,
        model: SpriteModel,
        minerals: Vec<UVec3>,
        spawn_pos: DVec3,
        angular_vel: DVec3,
    },
}

/// Pure-CPU chunk evaluation: density noise + optional asteroid model building.
/// No GPU or ECS access — safe to call from rayon threads.
/// `perlin` is constructed once per batch by the caller and shared across threads.
pub(crate) fn compute_chunk(
    chunk: IVec3,
    world_seed: u64,
    perlin: &PerlinNoise3D,
) -> ChunkComputeResult {
    let raw = perlin.fbm(
        chunk.x as f32,
        chunk.y as f32,
        chunk.z as f32,
        CHUNK_NOISE_OCTAVES,
        CHUNK_NOISE_FREQ,
    );
    let density = ((raw / PERLIN_MAX) + 1.0) * 0.5;
    let density = density.clamp(0.0, 1.0);
    let density = density * density * (3.0 - 2.0 * density);

    if chunk_spawn_hash(world_seed, chunk) >= density {
        return ChunkComputeResult::NoSpawn { chunk };
    }

    let h = chunk_hash_base(world_seed, chunk);
    let chunk_centre = (chunk.as_dvec3() + DVec3::splat(0.5)) * CHUNK_SIZE as f64;
    let spawn_pos = chunk_centre + chunk_spawn_offset(h);
    let has_crystals =
        (splitmix64(h.wrapping_add(8)) >> 40) as f32 / 16_777_216.0 < CRYSTAL_SPAWN_CHANCE;
    let noise_seed = h.wrapping_add(9);
    let scale_seed = h.wrapping_add(10);
    let (model, minerals) = build_asteroid(
        h.wrapping_add(6),
        h.wrapping_add(7),
        noise_seed,
        scale_seed,
        has_crystals,
    );
    let angular_vel = chunk_spawn_angular_vel(h);

    ChunkComputeResult::Spawn {
        chunk,
        model,
        minerals,
        spawn_pos,
        angular_vel,
    }
}

/// Combine world seed and chunk coordinates into one u64 base hash.
/// Each axis is multiplied by a distinct irrational-derived constant
/// (φ, π, e scaled to u64) so that chunks differing on only one axis
/// produce unrelated sums before the splitmix64 avalanche pass.
fn chunk_hash_base(world_seed: u64, chunk: IVec3) -> u64 {
    splitmix64(
        world_seed
            .wrapping_add((chunk.x as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .wrapping_add((chunk.y as u64).wrapping_mul(0x6c62272e07bb0142))
            .wrapping_add((chunk.z as u64).wrapping_mul(0x4d2c6dfc5ac42aad)),
    )
}

fn chunk_spawn_hash(world_seed: u64, chunk: IVec3) -> f32 {
    let h = chunk_hash_base(world_seed, chunk);
    // Top 24 bits → [0, 1)
    (h >> 40) as f32 / 16_777_216.0
}

fn chunk_spawn_offset(h: u64) -> DVec3 {
    // Axis indices 0/1/2: each splitmix64 call decorrelates adjacent seed values.
    DVec3::from([0u64, 1, 2].map(|i| hash_to_signed(splitmix64(h.wrapping_add(i)))))
        * SPAWN_SAFE_RANGE
}

fn chunk_spawn_angular_vel(h: u64) -> DVec3 {
    // Indices 3/4/5 are independent from the position offset indices 0/1/2.
    DVec3::from([3u64, 4, 5].map(|i| hash_to_signed(splitmix64(h.wrapping_add(i)))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{chunk_hash_base, chunk_spawn_angular_vel, chunk_spawn_hash, chunk_spawn_offset};

    #[test]
    fn world_to_chunk_origin() {
        assert_eq!(world_to_chunk(DVec3::ZERO), IVec3::ZERO);
    }

    #[test]
    fn world_to_chunk_positive() {
        // 65 world units into chunk 1
        assert_eq!(
            world_to_chunk(DVec3::new(65.0, 0.0, 0.0)),
            IVec3::new(1, 0, 0)
        );
    }

    #[test]
    fn world_to_chunk_negative() {
        // -1 world units is in chunk -1 (floor division)
        assert_eq!(
            world_to_chunk(DVec3::new(-1.0, 0.0, 0.0)),
            IVec3::new(-1, 0, 0)
        );
    }

    #[test]
    fn world_to_chunk_boundary() {
        // exactly at a boundary belongs to the higher chunk
        assert_eq!(
            world_to_chunk(DVec3::new(64.0, 0.0, 0.0)),
            IVec3::new(1, 0, 0)
        );
    }

    #[test]
    fn chunks_in_sphere_radius0_is_just_center() {
        let result: Vec<_> = chunks_in_sphere(IVec3::ZERO, 0).collect();
        assert_eq!(result, vec![IVec3::ZERO]);
    }

    #[test]
    fn chunks_in_sphere_radius1_count() {
        // unit sphere in 3D integer grid: center + 6 face neighbours = 7
        let count = chunks_in_sphere(IVec3::ZERO, 1).count();
        assert_eq!(count, 7);
    }

    #[test]
    fn chunks_in_sphere_all_within_radius() {
        let radius = 3;
        let r2 = radius * radius;
        for c in chunks_in_sphere(IVec3::ZERO, radius) {
            assert!(c.dot(c) <= r2, "{c} is outside radius {radius}");
        }
    }

    #[test]
    fn chunks_in_sphere_no_duplicates() {
        let seen: HashSet<IVec3> = chunks_in_sphere(IVec3::ZERO, 3).collect();
        let count = chunks_in_sphere(IVec3::ZERO, 3).count();
        assert_eq!(seen.len(), count);
    }

    #[test]
    fn missing_chunks_empty_visited_returns_full_sphere() {
        let visited = HashSet::new();
        let count = missing_chunks(DVec3::ZERO, 1, &visited).count();
        assert_eq!(count, 7);
    }

    #[test]
    fn missing_chunks_excludes_visited() {
        let center = IVec3::ZERO;
        let mut visited: HashSet<IVec3> = chunks_in_sphere(center, 1).collect();
        visited.remove(&IVec3::new(1, 0, 0));
        let missing: Vec<_> = missing_chunks(DVec3::ZERO, 1, &visited).collect();
        assert_eq!(missing, vec![IVec3::new(1, 0, 0)]);
    }

    #[test]
    fn missing_chunks_all_visited_returns_empty() {
        let visited: HashSet<IVec3> = chunks_in_sphere(IVec3::ZERO, 2).collect();
        assert!(missing_chunks(DVec3::ZERO, 2, &visited).next().is_none());
    }

    // ── chunk_spawn_hash ──────────────────────────────────────────────────────

    #[test]
    fn chunk_hash_in_unit_range() {
        for seed in [0u64, 1, u64::MAX, 0xdead_beef] {
            for chunk in [
                IVec3::ZERO,
                IVec3::new(1, -1, 1000),
                IVec3::new(-100, 200, -300),
            ] {
                let v = chunk_spawn_hash(seed, chunk);
                assert!(
                    (0.0..1.0).contains(&v),
                    "hash out of [0,1): {v} for chunk {chunk} seed {seed}"
                );
            }
        }
    }

    #[test]
    fn chunk_hash_differs_by_seed() {
        let chunk = IVec3::new(5, 5, 5);
        assert_ne!(
            chunk_spawn_hash(0, chunk),
            chunk_spawn_hash(1, chunk),
            "different seeds must produce different hashes"
        );
    }

    #[test]
    fn chunk_hash_differs_by_coord() {
        let seed = 42u64;
        let hx = chunk_spawn_hash(seed, IVec3::new(1, 0, 0));
        let hy = chunk_spawn_hash(seed, IVec3::new(0, 1, 0));
        let hz = chunk_spawn_hash(seed, IVec3::new(0, 0, 1));
        let hn = chunk_spawn_hash(seed, IVec3::new(-1, 0, 0));
        assert_ne!(hx, hy);
        assert_ne!(hy, hz);
        assert_ne!(hx, hn, "positive and negative coords must hash differently");
    }

    // ── chunk_spawn_angular_vel ───────────────────────────────────────────────

    #[test]
    fn spawn_angular_vel_in_unit_range() {
        for seed in [0u64, 1, 0xdead_beef, u64::MAX] {
            for chunk in [
                IVec3::ZERO,
                IVec3::new(1, -1, 1000),
                IVec3::new(-100, 200, -300),
            ] {
                let v = chunk_spawn_angular_vel(chunk_hash_base(seed, chunk));
                assert!(
                    v.abs().cmple(DVec3::ONE).all(),
                    "angular_vel {v} out of [-1,1] for chunk {chunk} seed {seed}"
                );
            }
        }
    }

    // ── chunk_spawn_offset ────────────────────────────────────────────────────

    #[test]
    fn spawn_offset_within_safe_range() {
        for seed in [0u64, 1, 0xdead_beef, u64::MAX] {
            for chunk in [
                IVec3::ZERO,
                IVec3::new(1, -1, 1000),
                IVec3::new(-100, 200, -300),
            ] {
                let o = chunk_spawn_offset(chunk_hash_base(seed, chunk));
                assert!(
                    o.abs().cmple(DVec3::splat(SPAWN_SAFE_RANGE)).all(),
                    "offset {o} exceeds safe range for chunk {chunk} seed {seed}"
                );
            }
        }
    }

    #[test]
    fn adjacent_asteroids_do_not_overlap() {
        // Worst case: both asteroids offset maximally toward the shared boundary.
        // Asteroid A in chunk (0,0,0) offset +SAFE in X; B in chunk (1,0,0) offset -SAFE in X.
        // Gap must be >= 2 * half_extent (16) for no AABB overlap.
        let half_extent = 8.0_f64;
        let chunk_a = IVec3::new(0, 0, 0);
        let chunk_b = IVec3::new(1, 0, 0);
        let centre_a = (chunk_a.as_dvec3() + DVec3::splat(0.5)) * CHUNK_SIZE as f64;
        let centre_b = (chunk_b.as_dvec3() + DVec3::splat(0.5)) * CHUNK_SIZE as f64;
        let worst_a = centre_a + DVec3::new(SPAWN_SAFE_RANGE, 0.0, 0.0);
        let worst_b = centre_b - DVec3::new(SPAWN_SAFE_RANGE, 0.0, 0.0);
        let gap = (worst_b.x - worst_a.x).abs();
        assert!(
            gap >= 2.0 * half_extent,
            "gap {gap} < min separation {}",
            2.0 * half_extent
        );
    }
}
