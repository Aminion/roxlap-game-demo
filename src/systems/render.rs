use glam::{DQuat, DVec3, Vec3};
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
    #[resource] perf: &mut PerformanceInfo,
    world: &SubWorld,
) {
    let screen_size = egui::vec2(screen.width as f32, screen.height as f32);
    let half = screen_size / 2.0;
    let fov_y_rad = screen.fov_y_rad;

    let core_cam = {
        let mut query = <&CameraComponent>::query();
        query
            .iter(world)
            .next()
            .expect("no CameraComponent entity")
            .0
    };

    let world_cam = GpuCamera {
        fov_y_rad,
        ..core_cam
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

    // Snapshot work time before vsync blocks inside render_scene.
    perf.work_time_us_raw = perf.work_timer.elapsed().as_micros() as u64;

    gpu.render_scene(
        &gpu_world.scene,
        &[world_cam, cube_cam],
        &world_cam,
        fov_y_rad,
        128,
    );

    // Project target_dir into screen space.
    // fov_y = 2*atan(h/w) → tan(fov_y/2) = h/w → focal_pixels = w/2.
    let target_screen = {
        let td = screen.target_dir.as_vec3();
        let f = td.dot(Vec3::from(world_cam.forward));
        if f > 0.01 {
            let r = td.dot(Vec3::from(world_cam.right));
            let d = td.dot(Vec3::from(world_cam.down));
            let focal = half.x;
            Some(egui::pos2(half.x + focal * r / f, half.y + focal * d / f))
        } else {
            None
        }
    };

    draw_hud(egui_ctx, gpu, screen_size, target_screen, perf);

    perf.work_timer = std::time::Instant::now();
}

fn draw_hud(
    egui_ctx: &egui::Context,
    gpu: &mut GpuRenderer,
    screen_size: egui::Vec2,
    target_screen: Option<egui::Pos2>,
    perf: &PerformanceInfo,
) {
    let half = screen_size / 2.0;

    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, screen_size)),
        ..Default::default()
    };

    let full_output = egui_ctx.run(raw_input, |ctx| {
        egui::Area::new(egui::Id::new("hud_perf"))
            .fixed_pos(egui::pos2(8.0, 8.0))
            .interactable(false)
            .show(ctx, |ui| {
                ui.colored_label(egui::Color32::YELLOW, format!("FPS {}", perf.fps));
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!("WORK   {:.2} ms", perf.work_time_us as f64 / 1000.0),
                );
            });

        let center = egui::pos2(half.x, half.y);
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("crosshair"),
        ));
        painter.circle_stroke(
            center,
            20.0,
            egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(255, 0, 255)),
        );
        if let Some(tp) = target_screen {
            painter.circle_stroke(
                tp,
                5.0,
                egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(255, 0, 255)),
            );
        }
    });

    let clipped_prims = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    gpu.paint_egui(
        &clipped_prims,
        &full_output.textures_delta,
        full_output.pixels_per_point,
    );
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
    let world_pos = Vec3::from(world_cam.position).as_dvec3();
    let pos = inv * (world_pos - cube_center) + vxl_center;
    let fwd = inv * Vec3::from(world_cam.forward).as_dvec3();
    let right = inv * Vec3::from(world_cam.right).as_dvec3();
    let down = inv * Vec3::from(world_cam.down).as_dvec3();
    GpuCamera {
        position: pos.as_vec3().into(),
        forward: fwd.as_vec3().into(),
        right: right.as_vec3().into(),
        down: down.as_vec3().into(),
        fov_y_rad,
    }
}
