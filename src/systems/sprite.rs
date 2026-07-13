use legion::{systems::CommandBuffer, world::SubWorld, Entity, EntityStore};
use roxlap_gpu::{SpriteModel, SpriteModelRegistry};
use roxlap_render::{DynSpriteTransform, Kv6, SceneRenderer, SpriteModelId, VoxColor};

use crate::components::sprite_id::Sprite;

/// Remove a single entity from the renderer, ECS, CPU registry, and all
/// bookkeeping.
///
/// Does NOT touch `VisitedChunks` — callers that want the chunk to be
/// re-populatable (distance-based unload) must remove it from `visited`
/// themselves.
pub fn perform_despawn(
    entity: Entity,
    world: &SubWorld,
    commands: &mut CommandBuffer,
    renderer: &mut SceneRenderer,
    registry: &mut SpriteModelRegistry,
) {
    let Ok(entry) = world.entry_ref(entity) else {
        return;
    };
    let Ok(sprite) = entry.get_component::<Sprite>() else {
        return;
    };
    renderer.remove_sprite_instance(sprite.instance_id);
    if sprite.owns_model {
        renderer.remove_sprite_model(sprite.model_id);
        // Free the CPU-side voxel data too — without this every asteroid
        // ever spawned stays resident in the registry for the whole session.
        registry.remove(sprite.chain_id);
    }
    commands.remove(entity);
}

/// Popcount-rank index into `model.colors` for the occupied voxel at
/// column `col` (occupancy block `base`), word `z_word`, bit `z_bit`.
pub fn voxel_color_index(
    model: &SpriteModel,
    col: usize,
    base: usize,
    z_word: usize,
    z_bit: u32,
) -> usize {
    let mut rank = model.color_offsets[col] as usize;
    for w in 0..z_word {
        rank += model.occupancy[base + w].count_ones() as usize;
    }
    let below_mask = (1u32 << z_bit) - 1;
    rank += (model.occupancy[base + z_word] & below_mask).count_ones() as usize;
    rank
}

/// Convert a dense-occupancy `SpriteModel` into a surface-only `Kv6` for the renderer.
pub fn sprite_model_to_kv6(model: &SpriteModel) -> Kv6 {
    let [mx, my, mz] = model.dims;
    let occ = model.occ_words_per_col as usize;
    Kv6::from_fn_shaded(mx, my, mz, |x, y, z| {
        let col = (x + y * mx) as usize;
        let base = col * occ;
        let z_word = z as usize / 32;
        let z_bit = z % 32;
        if (model.occupancy[base + z_word] >> z_bit) & 1 == 0 {
            return None;
        }
        let color_idx = voxel_color_index(model, col, base, z_word, z_bit);
        Some(VoxColor(model.colors[color_idx]))
    })
}

/// Spawn an additional instance of a pre-registered shared model (no model ownership).
pub fn spawn_shared_instance(
    renderer: &mut SceneRenderer,
    model_id: SpriteModelId,
    chain_id: u32,
) -> Sprite {
    let instance_id = renderer
        .add_sprite_instance_posed(model_id, DynSpriteTransform::default())
        .expect("shared sprite model is live");
    Sprite {
        chain_id,
        model_id,
        instance_id,
        owns_model: false,
    }
}

/// Register a sprite model with both the CPU registry and the renderer.
/// `kv6` must be the surface conversion of `model` (`sprite_model_to_kv6`),
/// taken precomputed so callers can run the conversion off the main thread.
pub fn spawn_sprite(
    renderer: &mut SceneRenderer,
    registry: &mut SpriteModelRegistry,
    model: SpriteModel,
    kv6: &Kv6,
) -> Sprite {
    let chain_id = registry.add(model);
    let model_id = renderer.add_sprite_model(kv6);
    let instance_id = renderer
        .add_sprite_instance_posed(model_id, DynSpriteTransform::default())
        .expect("freshly registered sprite model is live");
    Sprite {
        chain_id,
        model_id,
        instance_id,
        owns_model: true,
    }
}
