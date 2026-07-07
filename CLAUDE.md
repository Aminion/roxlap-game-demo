# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build
cargo run
cargo check
cargo clippy
cargo fmt
RUST_LOG=info cargo run
```

The dev environment uses Nix (`shell.nix`) with a nightly Rust toolchain and SDL2 libraries; on non-Nix systems SDL2, SDL2_mixer, SDL2_gfx, and SDL2_ttf must be installed separately.

## Architecture

Rust game demo using **Legion ECS**. Entry point: `src/main.rs`. Renderer: `SceneRenderer` (roxlap-render 0.25). Reference: https://ncrashed.github.io/roxlap/

### Coordinate conventions

- World space is **z-down**: small z = up, large z = down
- Body-local axes: **−Z = forward (nose)**, **+X = right**, **+Y = up**
- Camera uses `forward`, `right`, `down`; render system sets `down = −body_up`
- Asteroid voxel grid: `ASTEROID_VOXEL_SIZE = 16` (`src/sprites.rs`)

### Key invariants

- `NewtonBody.orientation` must stay unit-length; `integrate_rotation` normalizes every tick
- OBB→AABB: project half-extents through **rows** of |R|, not columns
- `from_mat3` requires det = +1; silently returns non-unit quat for left-handed input
- Thruster commands are in body space; autopilot converts via `orientation.inverse() *`

## Before committing

Run `cargo fmt` and `cargo test` before every commit.
