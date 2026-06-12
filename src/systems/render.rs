use glam::{DQuat, DVec3};
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_gpu::{camera::Camera as GpuCamera, GpuRenderer};

use crate::{
    components::{camera::CameraComponent, cube_marker::CubeMarker, newton_body::NewtonBody},
    GpuWorldData, ScreenState,
};

// --- Dead code kept for HUD migration ---
#[allow(dead_code, unused_imports)]
mod hud_dead {
    use crate::fonts::FontRenderer;
    use crate::systems::performance_info::PerformanceInfo;
    use crate::{CanvasResources, RenderBuffers, RenderTexture, ScreenState, Worlds};
    use roxlap_core::{
        opticast, scalar_rasterizer::ScalarRasterizer, Camera, GridView, OpticastSettings,
    };
    use roxlap_formats::vxl::Vxl;
    use sdl2::gfx::primitives::DrawRenderer;
    use sdl2::pixels::{Color, PixelFormatEnum};

    #[allow(dead_code)]
    fn run_opticast_pass(
        fb: &mut [u32],
        zb: &mut [f32],
        rw: u32,
        vxl: &Vxl,
        pool: &mut roxlap_core::rasterizer::ScratchPool,
        camera: &Camera,
        settings: &OpticastSettings,
    ) {
        let grid = GridView::from_single_vxl(vxl);
        let mut rasterizer = ScalarRasterizer::new(fb, zb, rw as usize, grid);
        let _ = opticast(&mut rasterizer, pool, camera, settings, grid);
    }

    #[allow(dead_code)]
    fn render_gui(
        canvas_resources: &mut CanvasResources,
        font_renderer: &FontRenderer,
        perf: &PerformanceInfo,
        window_w: u32,
        window_h: u32,
        target: (i32, i32, bool),
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
        let _ = canvas.circle(cx as i16, cy as i16, 20, Color::MAGENTA);
        let _ = canvas.circle(target.0 as i16, target.1 as i16, 5, Color::MAGENTA);
    }
}

// --- Active GPU render system ---

#[allow(clippy::too_many_arguments)]
#[system]
#[read_component(CameraComponent)]
#[read_component(CubeMarker)]
#[read_component(NewtonBody)]
pub fn render(
    #[resource] gpu: &mut GpuRenderer,
    #[resource] gpu_world: &GpuWorldData,
    #[resource] screen: &ScreenState,
    world: &SubWorld,
) {
    let (w, h) = (screen.width, screen.height);
    let fov_y_rad = 2.0 * f32::atan(h as f32 / w as f32);

    let core_cam = {
        let mut query = <&CameraComponent>::query();
        query
            .iter(world)
            .next()
            .expect("no CameraComponent entity")
            .0
    };

    let world_cam = GpuCamera {
        position: [
            core_cam.pos[0] as f32,
            core_cam.pos[1] as f32,
            core_cam.pos[2] as f32,
        ],
        forward: [
            core_cam.forward[0] as f32,
            core_cam.forward[1] as f32,
            core_cam.forward[2] as f32,
        ],
        right: [
            core_cam.right[0] as f32,
            core_cam.right[1] as f32,
            core_cam.right[2] as f32,
        ],
        down: [
            core_cam.down[0] as f32,
            core_cam.down[1] as f32,
            core_cam.down[2] as f32,
        ],
        fov_y_rad,
    };

    let cube_cam = {
        let mut q = <(&CubeMarker, &NewtonBody)>::query();
        q.iter(world)
            .next()
            .map(|(_, b)| {
                cube_space_gpu_cam(
                    &core_cam,
                    b.orientation,
                    b.pos,
                    crate::CUBE_VXL_VSID,
                    fov_y_rad,
                )
            })
            .unwrap_or(world_cam)
    };

    gpu.render_scene(
        &gpu_world.scene,
        &[world_cam, cube_cam],
        &world_cam,
        fov_y_rad,
        128,
    );
    gpu.present();
}

fn cube_space_gpu_cam(
    world_cam: &roxlap_core::Camera,
    orientation: DQuat,
    cube_center: DVec3,
    cube_vsid: u32,
    fov_y_rad: f32,
) -> GpuCamera {
    let vxl_center = DVec3::splat(f64::from(cube_vsid) / 2.0 - 0.5);
    let inv = orientation.inverse();
    let world_pos = DVec3::from(world_cam.pos);
    let pos = inv * (world_pos - cube_center) + vxl_center;
    let fwd = inv * DVec3::from(world_cam.forward);
    let right = inv * DVec3::from(world_cam.right);
    let down = inv * DVec3::from(world_cam.down);
    GpuCamera {
        position: [pos.x as f32, pos.y as f32, pos.z as f32],
        forward: [fwd.x as f32, fwd.y as f32, fwd.z as f32],
        right: [right.x as f32, right.y as f32, right.z as f32],
        down: [down.x as f32, down.y as f32, down.z as f32],
        fov_y_rad,
    }
}
