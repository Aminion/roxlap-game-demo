use std::time::Instant;

use glam::DVec3;
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_core::{
    opticast, scalar_rasterizer::ScalarRasterizer, Camera, Engine, GridView, OpticastSettings,
};
use roxlap_formats::vxl::Vxl;
use sdl2::pixels::{Color, PixelFormatEnum};

use crate::{
    components::{miner::Miner, newton_body::NewtonBody},
    fonts::FontRenderer,
    systems::performance_info::PerformanceInfo,
    CanvasResources, RenderBuffers, RenderTexture, WindowSize,
};

#[system]
#[read_component(Miner)]
#[read_component(NewtonBody)]
pub fn render(
    #[resource] canvas_resources: &mut CanvasResources,
    #[resource] world_map: &Vxl,
    #[resource] engine: &Engine,
    #[resource] render_tex: &mut RenderTexture,
    #[resource] buffers: &mut RenderBuffers,
    #[resource] window_size: &WindowSize,
    #[resource] font_renderer: &FontRenderer,
    #[resource] perf: &mut PerformanceInfo,
    world: &SubWorld,
) {
    let t_frame = Instant::now();

    let (w, h) = (window_size.0, window_size.1);

    // Recreate buffers and texture if the window was resized.
    if buffers.width != w || buffers.height != h {
        *buffers = RenderBuffers::new(w, h, crate::VSID);
        render_tex.0 = canvas_resources
            .texture_creator
            .create_texture_streaming(PixelFormatEnum::ARGB8888, w, h)
            .expect("resize texture failed");
    }

    // Push per-frame engine state onto the scratch pool (sky colour, side shades).
    let sky = engine.sky_color();
    let sky_i = i32::from_ne_bytes(sky.to_ne_bytes());
    buffers.pool.set_skycast(sky_i, 0);
    let s = engine.side_shades();
    buffers.pool.set_side_shades(s[0], s[1], s[2], s[3], s[4], s[5]);

    // Translate the miner's rigid-body state to a voxlap Camera.
    //
    // Body-local axis conventions (matching miner_input.rs):
    //   -Z = nose (forward), +X = right wing, +Y = top of body
    //
    // Rotating each body-local axis by `orientation` gives its world-space
    // direction. Voxlap's Camera wants `down` (not `up`), so we negate Y.
    let camera = {
        let mut query = <(&Miner, &NewtonBody)>::query();
        if let Some((_, body)) = query.iter(world).next() {
            let fwd = body.orientation * DVec3::NEG_Z;
            let right = body.orientation * DVec3::X;
            let up = body.orientation * DVec3::Y;
            Camera {
                pos: body.pos.to_array(),
                forward: fwd.to_array(),
                right: right.to_array(),
                down: (-up).to_array(),
            }
        } else {
            let cx = f64::from(crate::VSID) * 0.5;
            let cy = f64::from(crate::VSID) * 0.5;
            let cz = f64::from(crate::GROUND_Z) - f64::from(crate::CUBE_EDGE) - 6.0;
            let (sp, cp) = (0.15_f64.sin(), 0.15_f64.cos());
            Camera {
                pos: [cx - 16.0, cy, cz],
                forward: [cp, 0.0, sp],
                right: [0.0, 1.0, 0.0],
                down: [-sp, 0.0, cp],
            }
        }
    };

    buffers.framebuffer.fill(sky);

    // --- Phase 1: opticast (CPU ray-cast) ---
    let t_opticast = Instant::now();
    let settings = OpticastSettings::for_oracle_framebuffer(w, h);
    {
        let grid = GridView::from_single_vxl(world_map);
        let mut rasterizer = ScalarRasterizer::new(
            &mut buffers.framebuffer,
            &mut buffers.zbuffer,
            w as usize,
            grid,
        );
        let _ = opticast(&mut rasterizer, &mut buffers.pool, &camera, &settings, grid);
    }
    perf.opticast_us_raw = t_opticast.elapsed().as_micros() as u64;

    // --- Phase 2: SDL2 texture upload + blit ---
    let t_upload = Instant::now();
    render_tex
        .0
        .update(None, bytemuck::cast_slice(&buffers.framebuffer), (w * 4) as usize)
        .expect("texture update failed");
    canvas_resources.canvas.clear();
    canvas_resources.canvas.copy(&render_tex.0, None, None).unwrap();
    perf.upload_us_raw = t_upload.elapsed().as_micros() as u64;

    perf.frame_time_us_raw = t_frame.elapsed().as_micros() as u64;

    font_renderer.draw_text(
        &mut canvas_resources.canvas,
        &canvas_resources.texture_creator,
        &format!(
            "FPS {}\nFRAME  {:.2} ms\nOPTI   {:.2} ms\nUPLOAD {:.2} ms",
            perf.fps,
            perf.frame_time_us as f64 / 1000.0,
            perf.opticast_us as f64 / 1000.0,
            perf.upload_us as f64 / 1000.0,
        ),
        4,
        4,
        16.0,
        Color::YELLOW,
    );

    canvas_resources.canvas.present();
}
