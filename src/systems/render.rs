use bytemuck::Zeroable;
use glam::{DVec3, Vec3};
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_gpu::{
    camera::Camera as GpuCamera, GpuRenderer, SpriteInstance, SpriteInstanceTransform,
};

use crate::{
    components::{camera::CameraComponent, newton_body::NewtonBody, sprite_id::Sprite},
    systems::{
        energy::{Energy, ENERGY_LOW, ENERGY_MED},
        performance_info::PerformanceInfo,
    },
    AutopilotTarget, GpuWorldData, ScreenState,
};

#[allow(clippy::too_many_arguments)]
#[system]
#[read_component(CameraComponent)]
#[read_component(Sprite)]
#[read_component(NewtonBody)]
pub fn render(
    #[resource] gpu: &mut GpuRenderer,
    #[resource] gpu_world: &GpuWorldData,
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

    // Rebuild all asteroid sprite transforms each frame.
    {
        let count = gpu.sprite_instance_count();
        if count > 0 {
            let mut transforms: Vec<SpriteInstance> = vec![
                SpriteInstance {
                    model_id: 0,
                    transform: SpriteInstanceTransform::zeroed(),
                };
                count
            ];

            let mut q = <(&Sprite, &NewtonBody)>::query();
            for (sprite, b) in q.iter(world) {
                let slot = sprite.slot as usize;
                if slot < count {
                    transforms[slot] = sprite_from_body(sprite.chain_id, b);
                }
            }

            gpu.update_sprite_instance_transforms(&transforms);
        }
    }

    // Snapshot work time before vsync blocks inside render_scene.
    perf.work_time_us_raw = perf.work_timer.elapsed().as_micros() as u64;

    gpu.render_scene(&gpu_world.scene, &[], &world_cam, fov_y_rad, 128);

    // Project target_dir into screen space.
    // fov_y = 2*atan(h/w) → tan(fov_y/2) = h/w → focal_pixels = w/2.
    let target_screen = {
        let td = autopilot_target.0.as_vec3();
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

    draw_hud(egui_ctx, gpu, screen_size, target_screen, perf, energy);

    perf.work_timer = std::time::Instant::now();
}

fn draw_hud(
    egui_ctx: &egui::Context,
    gpu: &mut GpuRenderer,
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
    gpu.paint_egui(
        &clipped_prims,
        &full_output.textures_delta,
        full_output.pixels_per_point,
    );
}

pub(crate) fn sprite_from_body(chain_id: u32, b: &NewtonBody) -> SpriteInstance {
    let s = (b.orientation * DVec3::X).as_vec3();
    let h = (b.orientation * DVec3::Y).as_vec3();
    let f = (b.orientation * DVec3::Z).as_vec3();
    let mut transform = SpriteInstanceTransform::zeroed();
    transform.inv_rot = [
        [s.x, h.x, f.x, 0.0],
        [s.y, h.y, f.y, 0.0],
        [s.z, h.z, f.z, 0.0],
    ];
    transform.pos = b.pos.as_vec3().to_array();
    SpriteInstance {
        model_id: chain_id,
        transform,
    }
}
