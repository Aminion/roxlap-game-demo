# roxlap-game-demo

> **Demo project for the [roxlap](https://github.com/Aminion/roxlap) voxel engine.**
> Showcases real-time voxel rendering, physics-based flight, procedural asteroid generation, and projectile/crystal mechanics built on top of the roxlap GPU renderer.

---

## Gameplay

You pilot a mining ship through a procedurally generated asteroid field. Your ship runs on energy — thrusting, shooting, and using the retrieval beam all consume it. Crystals embedded in asteroids regenerate your energy when collected nearby.

**Objective:** shoot asteroids to expose crystals, retrieve crystals with the tractor beam, and keep your energy from hitting zero.

### Energy

| Indicator color | Meaning |
|-----------------|---------|
| Cyan | Energy OK (≥ 90) |
| Yellow | Low energy (30–90) |
| Red | Critical (< 30) |

When energy reaches 0 the game pauses and shows a **GAME OVER** screen. Press **Enter** to restart the world.

### Energy sources and costs

| Action | Energy effect |
|--------|--------------|
| Thruster firing | −5 per unit of effort per second |
| Cannon shot | −5 per shot |
| Retrieval beam | −5 per second |
| Crystal nearby (≤ 8 m) | +25 per crystal per second |

---

## Controls

### Flight

| Key | Action |
|-----|--------|
| W / S | Thrust up / down (ship +Y / −Y) |
| A / D | Thrust left / right (ship −X / +X) |
| LShift / Space | Thrust forward / backward (ship −Z / +Z) |
| Q / E | Roll counter-clockwise / clockwise |
| Tab (hold) | Retro-thrusters — damp all linear velocity and roll |

### Aiming and weapons

| Input | Action |
|-------|--------|
| Mouse | Move crosshair — sets autopilot aim direction |
| Left mouse button | Fire cannon (shoots projectiles at asteroids) |
| Right mouse button (hold) | Retrieval beam — pulls the nearest crystal in your crosshair toward you |

### System

| Key | Action |
|-----|--------|
| Escape | Quit |
| Enter | Restart (only when out of energy) |

---

## Autopilot

The ship continuously steers its nose toward the mouse crosshair using a bang-bang controller with a deceleration profile: it accelerates toward the target at full thrust, then begins braking early enough to stop without overshoot. Inside a small dead zone it switches to a PD controller for smooth settling.

Roll (Q / E) is independent — the autopilot ignores it and does not damp it.

---

## Building and running

The dev environment uses Nix (`shell.nix`) with a nightly Rust toolchain and SDL2 libraries. On non-Nix systems install SDL2, SDL2\_mixer, SDL2\_gfx, and SDL2\_ttf separately.

```bash
cargo run          # debug build
cargo run --release
```
