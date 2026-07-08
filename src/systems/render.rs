use glam::{DMat3, Vec3};
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_core::{opticast::OpticastSettings, Camera};
use roxlap_render::{
    DirectionalLight, DynSpriteTransform, FrameParams, LightRig, Line3, OverlayColor, Rgb,
    SceneRenderer,
};

use crate::{
    components::{camera::CameraComponent, newton_body::NewtonBody, sprite_id::Sprite},
    systems::lighting::{PointLights, SpotLights},
    RetrievalBeam, ScreenState,
};

#[system]
#[read_component(CameraComponent)]
#[read_component(Sprite)]
#[read_component(NewtonBody)]
pub fn render(
    #[resource] renderer: &mut SceneRenderer,
    #[resource] scene: &mut roxlap_scene::Scene,
    #[resource] screen: &ScreenState,
    #[resource] perf: &mut crate::systems::performance_info::PerformanceInfo,
    #[resource] point_lights: &PointLights,
    #[resource] spot_lights: &SpotLights,
    #[resource] beam: &RetrievalBeam,
    world: &SubWorld,
) {
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
            updates.push((sprite.instance_id, sprite_from_body(body, Vec3::ONE)));
        }
        renderer.set_sprite_instance_transforms(&updates);
    }

    // Snapshot work time before vsync blocks inside render.
    perf.work_time_us_raw = perf.work_timer.elapsed().as_micros() as u64;

    let settings =
        OpticastSettings::for_oracle_framebuffer(screen.width, screen.height).with_fov_y(fov_y_rad);
    let cam_fwd = camera.forward.map(|v| v as f32);
    let mut frame = FrameParams::new(&settings);
    frame.sky_color = Rgb(0);
    frame.fog_color = Rgb(0);
    frame.fog_max_scan_dist = 0;
    frame.treat_z_max_as_air = false;
    frame.draw_sprites = true;
    frame.side_shades = [0; 6];
    frame.lights = Some(LightRig {
        sun: Some(DirectionalLight {
            direction: cam_fwd,
            color: [1.0; 3],
            intensity: 0.25,
            casts_shadow: true,
        }),
        points: &point_lights.0,
        spots: &spot_lights.0,
        ambient: [0.25; 3],
        ..LightRig::default()
    });
    renderer.render(scene, &camera, &frame);

    // Overlay lines land in the framebuffer between render and paint_egui,
    // depth-tested against this frame's z-buffer.
    if let Some([a, b]) = beam.0 {
        renderer.draw_lines(
            &camera,
            &[Line3 {
                a: a.to_array(),
                b: b.to_array(),
                color: OverlayColor(0xB0_40E0FF), // translucent tractor-beam cyan
                width_px: 2.0,
                depth_test: true,
            }],
        );
    }
}

fn sprite_from_body(b: &NewtonBody, scale: Vec3) -> DynSpriteTransform {
    let rot = DMat3::from_quat(b.orientation);
    DynSpriteTransform {
        pos: b.pos.as_vec3().to_array(),
        right: (rot.x_axis * scale.x as f64).as_vec3().to_array(),
        up: (rot.y_axis * scale.y as f64).as_vec3().to_array(),
        forward: (rot.z_axis * scale.z as f64).as_vec3().to_array(),
    }
}
