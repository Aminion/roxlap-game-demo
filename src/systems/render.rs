use std::time::Instant;

use glam::DVec3;
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_core::{
    opticast, scalar_rasterizer::ScalarRasterizer, Camera, Engine, GridView, OpticastSettings,
};
use sdl2::pixels::{Color, PixelFormatEnum};

use crate::{
    components::{
        camera::CameraComponent, cube_marker::CubeMarker, newton_body::NewtonBody,
    },
    fonts::FontRenderer,
    systems::performance_info::PerformanceInfo,
    CanvasResources, RenderBuffers, RenderTexture, WindowSize, Worlds,
};

#[allow(clippy::too_many_arguments)]
#[system]
#[read_component(CameraComponent)]
#[read_component(CubeMarker)]
#[read_component(NewtonBody)]
pub fn render(
    #[resource] canvas_resources: &mut CanvasResources,
    #[resource] worlds: &Worlds,
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

    // Push per-frame engine state onto the scratch pool.
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

    let settings = OpticastSettings::for_oracle_framebuffer(w, h);

    // --- Pass 1: ground world ---
    buffers.framebuffer.fill(sky);
    let t_opticast = Instant::now();
    {
        let grid = GridView::from_single_vxl(&worlds.base);
        let mut rasterizer = ScalarRasterizer::new(
            &mut buffers.framebuffer,
            &mut buffers.zbuffer,
            w as usize,
            grid,
        );
        let _ = opticast(&mut rasterizer, &mut buffers.pool, camera, &settings, grid);
    }

    // --- Pass 2: rotating cube (camera-inverse-rotation) ---
    let cube_body = {
        let mut q = <(&CubeMarker, &NewtonBody)>::query();
        q.iter(world).next().map(|(_, b)| (b.orientation, b.pos))
    };
    if let Some((orientation, cube_center)) = cube_body {
        // VXL local center = (VSID/2 - 0.5) on each axis.
        let vxl_half = f64::from(crate::CUBE_VXL_VSID) / 2.0 - 0.5;
        let vxl_center = DVec3::splat(vxl_half);
        let inv = orientation.inverse();
        let world_pos = DVec3::from(camera.pos);

        let cube_cam = Camera {
            // Rotate camera into VXL local space around the cube's world center.
            // vxl_center is added AFTER rotation so the cube spins around its own center.
            pos: (inv * (world_pos - cube_center) + vxl_center).to_array(),
            forward: (inv * DVec3::from(camera.forward)).to_array(),
            right: (inv * DVec3::from(camera.right)).to_array(),
            down: (inv * DVec3::from(camera.down)).to_array(),
        };

        buffers.pool.set_skycast(sky_i, 0);
        buffers.pool.set_side_shades(s[0], s[1], s[2], s[3], s[4], s[5]);

        buffers.cube_fb.fill(sky);
        buffers.cube_zb.fill(0.0);
        {
            let grid = GridView::from_single_vxl(&worlds.cube);
            let mut rasterizer = ScalarRasterizer::new(
                &mut buffers.cube_fb,
                &mut buffers.cube_zb,
                w as usize,
                grid,
            );
            let _ = opticast(&mut rasterizer, &mut buffers.pool, &cube_cam, &settings, grid);
        }

        // Composite: cube geometry pixels over world.
        // Sky pixels in cube_fb equal `sky` (both pre-filled and written by rasterizer),
        // so checking != sky reliably identifies geometry hits.
        let n = (w * h) as usize;
        for i in 0..n {
            if buffers.cube_fb[i] != sky {
                buffers.framebuffer[i] = buffers.cube_fb[i];
            }
        }
    }

    perf.opticast_us_raw = t_opticast.elapsed().as_micros() as u64;

    // --- Phase 3: SDL2 texture upload + blit ---
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
