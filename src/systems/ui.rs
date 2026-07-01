use glam::Vec3;
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_render::SceneRenderer;

use crate::{
    components::camera::CameraComponent,
    systems::{
        energy::{Energy, ENERGY_LOW, ENERGY_MED},
        performance_info::PerformanceInfo,
    },
    AutopilotTarget, GameState, ScreenState,
};

#[system]
#[read_component(CameraComponent)]
pub fn ui(
    #[resource] renderer: &mut SceneRenderer,
    #[resource] egui_ctx: &egui::Context,
    #[resource] screen: &ScreenState,
    #[resource] game_state: &GameState,
    #[resource] autopilot_target: &AutopilotTarget,
    #[resource] energy: &Energy,
    #[resource] perf: &mut PerformanceInfo,
    world: &SubWorld,
) {
    let screen_size = egui::vec2(screen.width as f32, screen.height as f32);

    match game_state {
        GameState::TitleScreen => draw_controls_screen(egui_ctx, renderer, screen_size),
        GameState::GameOver => draw_game_over_screen(egui_ctx, renderer, screen_size),
        GameState::Playing => {
            let camera = {
                let mut q = <&CameraComponent>::query();
                q.iter(world).next().expect("no CameraComponent entity").0
            };
            let half = screen_size / 2.0;
            let td = autopilot_target.0.as_vec3();
            let fwd = Vec3::from_array(camera.forward.map(|v| v as f32));
            let target_screen = {
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
        }
    }

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

fn draw_game_over_screen(
    egui_ctx: &egui::Context,
    renderer: &mut SceneRenderer,
    screen_size: egui::Vec2,
) {
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, screen_size)),
        ..Default::default()
    };

    let full_output = egui_ctx.run_ui(raw_input, |ctx| {
        egui::Area::new(egui::Id::new("game_over"))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(210))
                    .inner_margin(egui::Margin::same(32_i8))
                    .show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.colored_label(
                                egui::Color32::RED,
                                egui::RichText::new("OUT OF ENERGY").size(48.0),
                            );
                            ui.add_space(16.0);
                            ui.colored_label(
                                egui::Color32::WHITE,
                                egui::RichText::new("PRESS ENTER TO RESTART").size(22.0),
                            );
                        });
                    });
            });
    });

    let clipped_prims = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    renderer.paint_egui(
        &clipped_prims,
        &full_output.textures_delta,
        full_output.pixels_per_point,
    );
}

fn draw_controls_screen(
    egui_ctx: &egui::Context,
    renderer: &mut SceneRenderer,
    screen_size: egui::Vec2,
) {
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, screen_size)),
        ..Default::default()
    };

    let full_output = egui_ctx.run_ui(raw_input, |ctx| {
        egui::Area::new(egui::Id::new("controls_guide"))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(210))
                    .inner_margin(egui::Margin::same(32_i8))
                    .show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.colored_label(
                                egui::Color32::WHITE,
                                egui::RichText::new("CONTROLS").size(28.0),
                            );
                        });
                        ui.add_space(16.0);
                        egui::Grid::new("controls_grid")
                            .spacing(egui::vec2(32.0, 6.0))
                            .show(ui, |ui| {
                                for (key, desc) in &[
                                    ("MOUSE", "Look / Aim"),
                                    ("LSHIFT", "Thrust Forward"),
                                    ("SPACE", "Thrust Backward"),
                                    ("W / S", "Thrust Up / Down"),
                                    ("A / D", "Thrust Left / Right"),
                                    ("Q / E", "Roll"),
                                    ("TAB", "Damping"),
                                    ("C", "Camera 1st/3rd"),
                                    ("LEFT CLICK", "Fire Cannon"),
                                    ("RIGHT CLICK", "Retrieve Crystals"),
                                    ("ESC", "Quit"),
                                ] {
                                    ui.colored_label(
                                        egui::Color32::YELLOW,
                                        egui::RichText::new(*key).size(16.0),
                                    );
                                    ui.colored_label(
                                        egui::Color32::LIGHT_GRAY,
                                        egui::RichText::new(*desc).size(16.0),
                                    );
                                    ui.end_row();
                                }
                            });
                        ui.add_space(20.0);
                        ui.vertical_centered(|ui| {
                            ui.colored_label(
                                egui::Color32::WHITE,
                                egui::RichText::new("PRESS ENTER TO START").size(22.0),
                            );
                        });
                    });
            });
    });

    let clipped_prims = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    renderer.paint_egui(
        &clipped_prims,
        &full_output.textures_delta,
        full_output.pixels_per_point,
    );
}
