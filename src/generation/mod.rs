pub mod chunks;

use std::f32::consts::{PI, TAU};

use glam::{IVec2, Vec2};
use rand::{
    distr::weighted::WeightedIndex, distr::Distribution, rngs::StdRng, RngExt, SeedableRng,
};

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
    let color_dist = WeightedIndex::new([2, 2, 6]).unwrap(); // 20% red, 20% blue, 60% white
    let size_dist = WeightedIndex::new([6, 3, 1]).unwrap(); // 60% sz2, 30% sz3, 10% sz4

    for _ in 0..STAR_COUNT {
        // Uniform solid-angle distribution: cos(polar) uniform in [-1, 1].
        let cos_theta: f32 = rng.random_range(-1.0f32..=1.0);
        let phi: f32 = rng.random_range(0.0f32..TAU);

        // Match the UV convention in scene_dda.wgsl:
        //   u = elevation = acos(-dir.z) / π  → texture column (x)
        //   v = azimuth  = atan2(x,y)/(2π)+.5 → texture row    (y)
        let uv = Vec2::new((-cos_theta).acos() / PI, phi / TAU);
        let center = IVec2::new(
            (uv.x * W as f32).min(W as f32 - 1.0) as i32,
            (uv.y * H as f32) as i32 % H as i32,
        );

        let brightness: u8 = rng.random_range(160u8..=220);
        let b = brightness as f32;

        // Color: 20% red-biased, 20% blue-biased, 60% white.
        let (r, g, bl) = match color_dist.sample(&mut rng) {
            0 => ((b * 1.0) as u8, (b * 0.80) as u8, (b * 0.75) as u8),
            1 => ((b * 0.75) as u8, (b * 0.85) as u8, (b * 1.0) as u8),
            _ => (brightness, brightness, brightness),
        };

        // Size: 2, 3, or 4 pixels — larger than before to survive bilinear blur.
        let size: i32 = match size_dist.sample(&mut rng) {
            0 => 2,
            1 => 3,
            _ => 4,
        };

        // In equirectangular, azimuth pixels compress by sin(elevation) near the poles.
        // Stretch the star in the azimuth (row) direction by 1/sin(elevation) so it
        // appears round on screen regardless of where on the sphere it sits.
        let sin_elev = (uv.x * PI).sin();
        let width_v = ((size as f32 / sin_elev).round() as i32).max(size);
        let half = IVec2::new(size / 2, width_v / 2);

        for dx in 0..size {
            // elevation (column) direction
            for dy in 0..width_v {
                // azimuth (row) direction — stretched
                let pixel = IVec2::new(
                    (center.x + dx - half.x).clamp(0, W as i32 - 1),
                    (center.y + dy - half.y).rem_euclid(H as i32),
                );
                let i = ((pixel.y as u32 * W + pixel.x as u32) * 4) as usize;
                pixels[i] = r;
                pixels[i + 1] = g;
                pixels[i + 2] = bl;
                pixels[i + 3] = 255;
            }
        }
    }

    (pixels, W, H)
}
