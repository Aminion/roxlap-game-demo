use glam::{DMat3, Vec3};
use legion::{system, world::SubWorld, IntoQuery};
use roxlap_core::{opticast::OpticastSettings, Camera};
use roxlap_render::{DirectionalLight, DynSpriteTransform, FrameParams, LightRig, SceneRenderer};

use crate::{
    components::{
        camera::CameraComponent, newton_body::NewtonBody, particle::Particle, sprite_id::Sprite,
    },
    systems::lighting::{PointLights, SpotLights},
    ScreenState,
};

#[system]
#[read_component(CameraComponent)]
#[read_component(Sprite)]
#[read_component(NewtonBody)]
#[read_component(Particle)]
pub fn render(
    #[resource] renderer: &mut SceneRenderer,
    #[resource] scene: &mut roxlap_scene::Scene,
    #[resource] screen: &ScreenState,
    #[resource] perf: &mut crate::systems::performance_info::PerformanceInfo,
    #[resource] point_lights: &PointLights,
    #[resource] spot_lights: &SpotLights,
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
        let mut q = <(&Sprite, &NewtonBody, Option<&Particle>)>::query();
        for (sprite, body, particle) in q.iter(world) {
            let scale = particle.map(|p| p.scale).unwrap_or(Vec3::ONE);
            updates.push((sprite.instance_id, sprite_from_body(body, scale)));
        }
        renderer.set_sprite_instance_transforms(&updates);
    }

    // Snapshot work time before vsync blocks inside render.
    perf.work_time_us_raw = perf.work_timer.elapsed().as_micros() as u64;

    let settings = OpticastSettings::for_oracle_framebuffer(screen.width, screen.height);
    let cam_fwd = camera.forward.map(|v| v as f32);
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
        lights: Some(LightRig {
            sun: Some(DirectionalLight {
                direction: cam_fwd,
                color: [1.0; 3],
                intensity: 0.25,
                casts_shadow: false,
            }),
            points: &point_lights.0,
            spots: &spot_lights.0,
            ambient: [0.25; 3],
            ..LightRig::default()
        }),
    };
    renderer.render(scene, &camera, &frame);
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
