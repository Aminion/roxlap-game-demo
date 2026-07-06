use glam::Vec3;
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_render::{ParticleSystem, SceneRenderer};

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
    #[resource] particle_sys: &ParticleSystem,
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
            let target_screen = project_target(
                autopilot_target.0.as_vec3(),
                Vec3::from_array(camera.forward.map(|v| v as f32)),
                Vec3::from_array(camera.right.map(|v| v as f32)),
                Vec3::from_array(camera.down.map(|v| v as f32)),
                half,
            );
            draw_hud(
                egui_ctx,
                renderer,
                screen_size,
                target_screen,
                perf,
                energy,
                particle_sys.particle_count(),
            );
        }
    }

    perf.work_timer = std::time::Instant::now();
}

/// Projects a world-space direction vector onto screen space.
///
/// `td` is the target direction relative to the camera origin (not normalised).
/// Returns `None` when the target is behind or near-perpendicular to the camera.
/// Focal length is set to `half.x` so the field-of-view is square horizontally.
fn project_target(
    td: Vec3,
    fwd: Vec3,
    right: Vec3,
    down: Vec3,
    half: egui::Vec2,
) -> Option<egui::Pos2> {
    let f = td.dot(fwd);
    if f > 0.01 {
        let r = td.dot(right);
        let d = td.dot(down);
        Some(egui::pos2(half.x + half.x * r / f, half.y + half.x * d / f))
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_hud(
    egui_ctx: &egui::Context,
    renderer: &mut SceneRenderer,
    screen_size: egui::Vec2,
    target_screen: Option<egui::Pos2>,
    perf: &PerformanceInfo,
    energy: &Energy,
    particle_count: usize,
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
                ui.colored_label(egui::Color32::YELLOW, format!("PART   {particle_count}"));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn half() -> egui::Vec2 {
        egui::vec2(400.0, 300.0)
    }

    #[test]
    fn project_target_on_axis_lands_at_center() {
        // Target straight along the forward vector → screen centre.
        let p = project_target(Vec3::NEG_Z, Vec3::NEG_Z, Vec3::X, Vec3::NEG_Y, half()).unwrap();
        assert!((p.x - 400.0).abs() < 1e-4, "x={}", p.x);
        assert!((p.y - 300.0).abs() < 1e-4, "y={}", p.y);
    }

    #[test]
    fn project_target_behind_camera_returns_none() {
        assert!(project_target(Vec3::Z, Vec3::NEG_Z, Vec3::X, Vec3::NEG_Y, half()).is_none());
    }

    #[test]
    fn project_target_45_degrees_right() {
        // Camera: fwd=+Z, right=+X, down=+Y.
        // Target at 45° right: dir = normalize(Z+X) → f = r = 1/√2, d = 0.
        // screen_x = half.x + half.x * (r/f) = 400 + 400*1 = 800, screen_y = 300.
        let dir = Vec3::new(1.0, 0.0, 1.0).normalize();
        let p = project_target(dir, Vec3::Z, Vec3::X, Vec3::Y, half()).unwrap();
        assert!((p.x - 800.0).abs() < 1e-3, "x={}", p.x);
        assert!((p.y - 300.0).abs() < 1e-3, "y={}", p.y);
    }

    #[test]
    fn project_target_depth_scales_offset() {
        // Doubling the distance halves the angular offset → same screen position.
        let dir_near = Vec3::new(1.0, 0.0, 2.0); // closer
        let dir_far = Vec3::new(2.0, 0.0, 4.0); // same direction, twice as far
        let p_near = project_target(dir_near, Vec3::Z, Vec3::X, Vec3::Y, half()).unwrap();
        let p_far = project_target(dir_far, Vec3::Z, Vec3::X, Vec3::Y, half()).unwrap();
        assert!(
            (p_near.x - p_far.x).abs() < 1e-4,
            "x should be equal: {} vs {}",
            p_near.x,
            p_far.x
        );
        assert!(
            (p_near.y - p_far.y).abs() < 1e-4,
            "y should be equal: {} vs {}",
            p_near.y,
            p_far.y
        );
    }
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
