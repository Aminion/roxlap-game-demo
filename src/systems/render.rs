use glam::{DVec3, Vec3};
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_core::{opticast::OpticastSettings, Camera};
use roxlap_render::{DynSpriteTransform, FrameParams, SceneRenderer};

use crate::{
    components::{camera::CameraComponent, newton_body::NewtonBody, sprite_id::Sprite},
    systems::{
        energy::{Energy, ENERGY_LOW, ENERGY_MED},
        performance_info::PerformanceInfo,
    },
    AutopilotTarget, ScreenState,
};

#[allow(clippy::too_many_arguments)]
#[system]
#[read_component(CameraComponent)]
#[read_component(Sprite)]
#[read_component(NewtonBody)]
pub fn render(
    #[resource] renderer: &mut SceneRenderer,
    #[resource] scene: &mut roxlap_scene::Scene,
    #[resource] screen: &ScreenState,
    #[resource] autopilot_target: &AutopilotTarget,
    #[resource] egui_ctx: &egui::Context,
    #[resource] perf: &mut PerformanceInfo,
    #[resource] energy: &Energy,
    world: &SubWorld,
) {
    let screen_size = egui::vec2(screen.width as f32, screen.height as f32);
    let half = screen_size / 2.0;
    let fov_y_rad = screen.fov_y_rad;

    let camera: Camera = {
        let mut query = <&CameraComponent>::query();
        query
            .iter(world)
            .next()
            .expect("no CameraComponent entity")
            .0
    };

    // Update all sprite instance transforms for this frame.
    {
        let mut updates: Vec<(roxlap_render::SpriteInstanceId, DynSpriteTransform)> = Vec::new();
        let mut q = <(&Sprite, &NewtonBody)>::query();
        for (sprite, body) in q.iter(world) {
            updates.push((sprite.instance_id, sprite_from_body(body)));
        }
        renderer.set_sprite_instance_transforms(&updates);
    }

    // Snapshot work time before vsync blocks inside render.
    perf.work_time_us_raw = perf.work_timer.elapsed().as_micros() as u64;

    let settings = OpticastSettings::for_oracle_framebuffer(screen.width, screen.height);
    let frame = FrameParams {
        settings: &settings,
        sky_color: 0,
        sky: None,
        fog_color: 0,
        fog_max_scan_dist: 0,
        treat_z_max_as_air: false,
        gpu_mip_scan_dist: 128.0,
        gpu_max_outer_steps: 128,
        gpu_fov_y_rad: fov_y_rad,
        draw_sprites: true,
        side_shades: [0; 6],
    };
    renderer.render(scene, &camera, &frame);

    // Project target_dir into screen space.
    let target_screen = {
        let td = autopilot_target.0.as_vec3();
        let fwd = Vec3::from_array(camera.forward.map(|v| v as f32));
        let f = td.dot(fwd);
        if f > 0.01 {
            let r = td.dot(Vec3::from_array(camera.right.map(|v| v as f32)));
            let d = td.dot(Vec3::from_array(camera.down.map(|v| v as f32)));
            let focal = half.x;
            Some(egui::pos2(half.x + focal * r / f, half.y + focal * d / f))
        } else {
            None
        }
    };

    draw_hud(egui_ctx, renderer, screen_size, target_screen, perf, energy);

    perf.work_timer = std::time::Instant::now();
}

fn draw_hud(
    egui_ctx: &egui::Context,
    renderer: &mut SceneRenderer,
    screen_size: egui::Vec2,
    target_screen: Option<egui::Pos2>,
    perf: &PerformanceInfo,
    energy: &Energy,
) {
    let half = screen_size / 2.0;

    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, screen_size)),
        ..Default::default()
    };

    let full_output = egui_ctx.run_ui(raw_input, |ctx| {
        egui::Area::new(egui::Id::new("hud_perf"))
            .fixed_pos(egui::pos2(8.0, 8.0))
            .interactable(false)
            .show(ctx, |ui| {
                ui.colored_label(egui::Color32::YELLOW, format!("FPS {}", perf.fps));
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!("WORK   {:.2} ms", perf.work_time_us_display as f64 / 1000.0),
                );
            });

        egui::Area::new(egui::Id::new("hud_energy"))
            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -12.0))
            .interactable(false)
            .show(ctx, |ui| {
                let color = if energy.current < ENERGY_LOW {
                    egui::Color32::RED
                } else if energy.current < ENERGY_MED {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::CYAN
                };
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(format!("ENERGY  {:.0}", energy.current))
                            .size(32.0)
                            .color(color),
                    )
                    .wrap_mode(egui::TextWrapMode::Extend),
                );
            });

        if energy.current <= 0.0 {
            egui::Area::new(egui::Id::new("out_of_energy"))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .interactable(false)
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.colored_label(
                            egui::Color32::RED,
                            egui::RichText::new("OUT OF ENERGY").size(48.0),
                        );
                        ui.add_space(8.0);
                        ui.colored_label(
                            egui::Color32::WHITE,
                            egui::RichText::new("PRESS ENTER TO RESTART").size(22.0),
                        );
                    });
                });
        }

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
    renderer.paint_egui(
        &clipped_prims,
        &full_output.textures_delta,
        full_output.pixels_per_point,
    );
}

fn sprite_from_body(b: &NewtonBody) -> DynSpriteTransform {
    let right = (b.orientation * DVec3::X).as_vec3();
    let up = (b.orientation * DVec3::Y).as_vec3();
    let forward = (b.orientation * DVec3::Z).as_vec3();
    DynSpriteTransform {
        pos: b.pos.as_vec3().to_array(),
        right: right.to_array(),
        up: up.to_array(),
        forward: forward.to_array(),
    }
}
