use std::time::Instant;

use glam::DVec3;
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_core::{
    opticast, scalar_rasterizer::ScalarRasterizer, Engine, GridView, OpticastSettings,
};
use roxlap_formats::edit::set_cube;
use sdl2::pixels::{Color, PixelFormatEnum};

use crate::{
    components::{
        camera::CameraComponent, cube_marker::CubeMarker, newton_body::NewtonBody,
    },
    fonts::FontRenderer,
    systems::performance_info::PerformanceInfo,
    BaseWorld, CanvasResources, RenderBuffers, RenderTexture, WindowSize,
};

#[allow(clippy::too_many_arguments)]
#[system]
#[read_component(CameraComponent)]
#[read_component(CubeMarker)]
#[read_component(NewtonBody)]
pub fn render(
    #[resource] canvas_resources: &mut CanvasResources,
    #[resource] base_world: &BaseWorld,
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
    buffers
        .pool
        .set_side_shades(s[0], s[1], s[2], s[3], s[4], s[5]);

    let camera = {
        let mut query = <&CameraComponent>::query();
        &query
            .iter(world)
            .next()
            .expect("no CameraComponent entity")
            .0
    };

    buffers.framebuffer.fill(sky);

    // Clone base world and stamp the rotating cube into it each frame.
    let cube_body = {
        let mut q = <(&CubeMarker, &NewtonBody)>::query();
        q.iter(world).next().map(|(_, b)| (b.orientation, b.pos))
    };
    let mut frame_world = base_world.0.clone();
    if let Some((orientation, center)) = cube_body {
        for lx in 0..crate::CUBE_EDGE {
            for ly in 0..crate::CUBE_EDGE {
                for lz in 0..crate::CUBE_EDGE {
                    let local = DVec3::new(
                        lx as f64 - f64::from(crate::CUBE_EDGE) / 2.0 + 0.5,
                        ly as f64 - f64::from(crate::CUBE_EDGE) / 2.0 + 0.5,
                        lz as f64 - f64::from(crate::CUBE_EDGE) / 2.0 + 0.5,
                    );
                    let wp = center + orientation * local;
                    set_cube(
                        &mut frame_world,
                        wp.x.round() as i32,
                        wp.y.round() as i32,
                        wp.z.round() as i32,
                        Some(crate::CUBE_COL),
                    );
                }
            }
        }
    }

    // --- Phase 1: opticast (CPU ray-cast) ---
    let t_opticast = Instant::now();
    let settings = OpticastSettings::for_oracle_framebuffer(w, h);
    {
        let grid = GridView::from_single_vxl(&frame_world);
        let mut rasterizer = ScalarRasterizer::new(
            &mut buffers.framebuffer,
            &mut buffers.zbuffer,
            w as usize,
            grid,
        );
        let _ = opticast(&mut rasterizer, &mut buffers.pool, camera, &settings, grid);
    }
    perf.opticast_us_raw = t_opticast.elapsed().as_micros() as u64;

    // --- Phase 2: SDL2 texture upload + blit ---
    let t_upload = Instant::now();
    render_tex
        .0
        .update(
            None,
            bytemuck::cast_slice(&buffers.framebuffer),
            (w * 4) as usize,
        )
        .expect("texture update failed");
    canvas_resources.canvas.clear();
    canvas_resources
        .canvas
        .copy(&render_tex.0, None, None)
        .unwrap();
    perf.upload_us_raw = t_upload.elapsed().as_micros() as u64;

    perf.frame_time_us_raw = t_frame.elapsed().as_micros() as u64;

    render_gui(canvas_resources, font_renderer, perf);

    canvas_resources.canvas.present();
}

fn render_gui(
    canvas_resources: &mut CanvasResources,
    font_renderer: &FontRenderer,
    perf: &PerformanceInfo,
) {
    font_renderer.draw_text(
        canvas_resources,
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
}
