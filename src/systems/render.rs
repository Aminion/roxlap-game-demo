use glam::{DQuat, DVec3};
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_gpu::{camera::Camera as GpuCamera, GpuRenderer};

use crate::{
    components::{camera::CameraComponent, cube_marker::CubeMarker, newton_body::NewtonBody},
    systems::performance_info::PerformanceInfo,
    GpuWorldData, ScreenState,
};

#[allow(clippy::too_many_arguments)]
#[system]
#[read_component(CameraComponent)]
#[read_component(CubeMarker)]
#[read_component(NewtonBody)]
pub fn render(
    #[resource] gpu: &mut GpuRenderer,
    #[resource] gpu_world: &GpuWorldData,
    #[resource] screen: &ScreenState,
    #[resource] egui_ctx: &egui::Context,
    #[resource] perf: &PerformanceInfo,
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

    let world_cam = GpuCamera { fov_y_rad, ..core_cam };

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

    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(w as f32, h as f32),
        )),
        ..Default::default()
    };

    let full_output = egui_ctx.run(raw_input, |ctx| {
        egui::Area::new(egui::Id::new("hud_perf"))
            .fixed_pos(egui::pos2(8.0, 8.0))
            .interactable(false)
            .show(ctx, |ui| {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!("FPS {}", perf.fps),
                );
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!("FRAME  {:.2} ms", perf.frame_time_us as f64 / 1000.0),
                );
            });

        let center = egui::pos2(w as f32 / 2.0, h as f32 / 2.0);
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("crosshair"),
        ));
        painter.circle_stroke(
            center,
            20.0,
            egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(255, 0, 255)),
        );
    });

    let clipped_prims = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    gpu.paint_egui(&clipped_prims, &full_output.textures_delta, full_output.pixels_per_point);
}

fn cube_space_gpu_cam(
    world_cam: &GpuCamera,
    orientation: DQuat,
    cube_center: DVec3,
    cube_vsid: u32,
    fov_y_rad: f32,
) -> GpuCamera {
    let vxl_center = DVec3::splat(f64::from(cube_vsid) / 2.0 - 0.5);
    let inv = orientation.inverse();
    let world_pos = DVec3::from(world_cam.position.map(f64::from));
    let pos = inv * (world_pos - cube_center) + vxl_center;
    let fwd = inv * DVec3::from(world_cam.forward.map(f64::from));
    let right = inv * DVec3::from(world_cam.right.map(f64::from));
    let down = inv * DVec3::from(world_cam.down.map(f64::from));
    GpuCamera {
        position: [pos.x as f32, pos.y as f32, pos.z as f32],
        forward: [fwd.x as f32, fwd.y as f32, fwd.z as f32],
        right: [right.x as f32, right.y as f32, right.z as f32],
        down: [down.x as f32, down.y as f32, down.z as f32],
        fov_y_rad,
    }
}
