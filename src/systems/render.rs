use std::time::Instant;

use glam::{DQuat, DVec3};
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_core::{
    opticast, rasterizer::ScratchPool, scalar_rasterizer::ScalarRasterizer, Camera, Engine,
    GridView, OpticastSettings,
};
use roxlap_formats::vxl::Vxl;
use sdl2::gfx::primitives::DrawRenderer;
use sdl2::pixels::{Color, PixelFormatEnum};

use crate::{
    components::{camera::CameraComponent, cube_marker::CubeMarker, newton_body::NewtonBody},
    fonts::FontRenderer,
    systems::performance_info::PerformanceInfo,
    CanvasResources, RenderBuffers, RenderTexture, ScreenState, Worlds,
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
    #[resource] screen: &ScreenState,
    #[resource] font_renderer: &FontRenderer,
    #[resource] perf: &mut PerformanceInfo,
    world: &SubWorld,
) {
    let t_frame = Instant::now();

    let (w, h) = (screen.width, screen.height);
    let (rw, rh) = ((w / 2).max(1), (h / 2).max(1));

    // Recreate buffers and texture if the window was resized.
    if buffers.width != rw || buffers.height != rh {
        *buffers = RenderBuffers::new(rw, rh, crate::VSID);
        render_tex.0 = canvas_resources
            .texture_creator
            .create_texture_streaming(PixelFormatEnum::ARGB8888, rw, rh)
            .expect("resize texture failed");
    }

    // Push per-frame engine state onto the scratch pool.
    let sky = engine.sky_color();
    buffers.pool.set_skycast(bytemuck::cast(sky), 0);
    let [s0, s1, s2, s3, s4, s5] = engine.side_shades();
    buffers.pool.set_side_shades(s0, s1, s2, s3, s4, s5);

    let camera = {
        let mut query = <&CameraComponent>::query();
        &query
            .iter(world)
            .next()
            .expect("no CameraComponent entity")
            .0
    };

    let settings = OpticastSettings::for_oracle_framebuffer(rw, rh);

    // --- Pass 1: ground world ---
    buffers.framebuffer.fill(sky);
    let t_opticast = Instant::now();
    run_opticast_pass(
        &mut buffers.framebuffer,
        &mut buffers.zbuffer,
        rw,
        &worlds.base,
        &mut buffers.pool,
        camera,
        &settings,
    );

    // --- Pass 2: rotating cube (camera-inverse-rotation) ---
    let cube_body = {
        let mut q = <(&CubeMarker, &NewtonBody)>::query();
        q.iter(world).next().map(|(_, b)| (b.orientation, b.pos))
    };
    if let Some((orientation, cube_center)) = cube_body {
        let cube_cam = cube_space_camera(camera, orientation, cube_center, crate::CUBE_VXL_VSID);

        buffers.cube_fb.fill(sky);
        buffers.cube_zb.fill(0.0);
        run_opticast_pass(
            &mut buffers.cube_fb,
            &mut buffers.cube_zb,
            rw,
            &worlds.cube,
            &mut buffers.pool,
            &cube_cam,
            &settings,
        );

        // Composite: cube geometry pixels over world.
        // Sky pixels in cube_fb equal `sky` (both pre-filled and written by rasterizer),
        // so checking != sky reliably identifies geometry hits.
        for (dst, &src) in buffers.framebuffer.iter_mut().zip(buffers.cube_fb.iter()) {
            if src != sky {
                *dst = src;
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
            (rw * 4) as usize,
        )
        .expect("texture update failed");
    canvas_resources.canvas.clear();
    canvas_resources
        .canvas
        .copy(&render_tex.0, None, None)
        .unwrap();
    perf.upload_us_raw = t_upload.elapsed().as_micros() as u64;

    perf.frame_time_us_raw = t_frame.elapsed().as_micros() as u64;

    // Project the world-space target direction onto the screen.
    // course (ship nose) is always screen center since the camera follows orientation.
    let target_screen = {
        let td = screen.target_dir;
        let cam_fwd = glam::DVec3::from(camera.forward);
        let cam_right = glam::DVec3::from(camera.right);
        let cam_down = glam::DVec3::from(camera.down);
        let tf = td.dot(cam_fwd);
        let tr = td.dot(cam_right);
        let tdown = td.dot(cam_down);
        // Scale factors match the opticast render half-extents so the indicator
        // stays in sync with the 3-D projection (wider H-FOV on 16:9 displays).
        let rw_f = (w / 2) as f64;
        let rh_f = (h / 2) as f64;
        let cx_f = w as f64 / 2.0;
        let cy_f = h as f64 / 2.0;
        if tf > 0.01 {
            let sx = (cx_f + tr / tf * rw_f).round() as i32;
            let sy = (cy_f + tdown / tf * rh_f).round() as i32;
            (sx.clamp(8, w as i32 - 8), sy.clamp(8, h as i32 - 8), false)
        } else {
            // Target is behind camera — push to screen edge.
            let scale = rw_f.max(rh_f) * 2.0;
            let sx = (cx_f + tr * scale).round() as i32;
            let sy = (cy_f + tdown * scale).round() as i32;
            (sx.clamp(8, w as i32 - 8), sy.clamp(8, h as i32 - 8), true)
        }
    };

    render_gui(canvas_resources, font_renderer, perf, w, h, target_screen);

    canvas_resources.canvas.present();
}

fn run_opticast_pass(
    fb: &mut [u32],
    zb: &mut [f32],
    rw: u32,
    vxl: &Vxl,
    pool: &mut ScratchPool,
    camera: &Camera,
    settings: &OpticastSettings,
) {
    let grid = GridView::from_single_vxl(vxl);
    let mut rasterizer = ScalarRasterizer::new(fb, zb, rw as usize, grid);
    let _ = opticast(&mut rasterizer, pool, camera, settings, grid);
}

fn cube_space_camera(
    world_cam: &Camera,
    orientation: DQuat,
    cube_center: DVec3,
    cube_vsid: u32,
) -> Camera {
    let vxl_center = DVec3::splat(f64::from(cube_vsid) / 2.0 - 0.5);
    let inv = orientation.inverse();
    let world_pos = DVec3::from(world_cam.pos);
    Camera {
        // vxl_center is added AFTER rotation so the cube spins around its own center.
        pos: (inv * (world_pos - cube_center) + vxl_center).to_array(),
        forward: (inv * DVec3::from(world_cam.forward)).to_array(),
        right: (inv * DVec3::from(world_cam.right)).to_array(),
        down: (inv * DVec3::from(world_cam.down)).to_array(),
    }
}

fn render_gui(
    canvas_resources: &mut CanvasResources,
    font_renderer: &FontRenderer,
    perf: &PerformanceInfo,
    window_w: u32,
    window_h: u32,
    target: (i32, i32, bool), // projected target_dir; bool = behind camera
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

    let canvas = &mut canvas_resources.canvas;
    let cx = (window_w / 2) as i32;
    let cy = (window_h / 2) as i32;

    // Course indicator — ship nose, always at screen center (camera follows orientation).
    let _ = canvas.circle(cx as i16, cy as i16, 20, Color::MAGENTA);

    // Target indicator — where the autopilot is steering toward.
    let _ = canvas.circle(target.0 as i16, target.1 as i16, 5, Color::MAGENTA);
}
