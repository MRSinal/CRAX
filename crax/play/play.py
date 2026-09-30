#!/usr/bin/env python3
"""Play a CRAX environment with the keyboard.

The environment is stepped with the regular (jitted) JAX `reset`/`step`, the
resulting MJX state is copied into a CPU `MjData` and rendered with MuJoCo's
offscreen renderer into a pygame window.

Usage:
    python -m crax.play
    python -m crax.play --env_name safe_push_point --level 3
    python -m crax.play --camera fixedfar --scale 1.5

Controls (point agent):
    W / Up       thrust forward        S / Down     thrust backward
    A / Left     turn left             D / Right    turn right
    C            cycle camera          R            reset episode
    Space        pause / resume        Esc / Q      quit
"""
import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

# Offscreen rendering; pygame owns the on-screen window. Override with MUJOCO_GL.
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import mujoco
import numpy as np

try:
    import pygame
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError("Playing CRAX requires pygame: `pip install pygame`") from e

from crax import envs


@dataclass(frozen=True)
class AgentControls:
    """Maps the currently held keys to an action vector for one agent type."""
    action_fn: Callable[[Sequence[bool]], np.ndarray]
    help_lines: List[str]
    cameras: List[str]


def _axis(keys: Sequence[bool], positive: Sequence[int], negative: Sequence[int]) -> float:
    return float(any(keys[k] for k in positive)) - float(any(keys[k] for k in negative))


def _point_action(keys: Sequence[bool]) -> np.ndarray:
    # Point actuators: [0] thrust along the agent's heading, [1] yaw velocity (+ = CCW / left).
    thrust = _axis(keys, (pygame.K_w, pygame.K_UP), (pygame.K_s, pygame.K_DOWN))
    turn = _axis(keys, (pygame.K_a, pygame.K_LEFT), (pygame.K_d, pygame.K_RIGHT))
    return np.array([thrust, turn], dtype=np.float32)


POINT_CONTROLS = AgentControls(
    action_fn=_point_action,
    help_lines=["W/S: thrust", "A/D: turn"],
    cameras=["track", "fixedfar", "vision"],
)

# Agent suffix of the env name (`safe_<task>_<agent>`) -> controls.
AGENT_CONTROLS: Dict[str, AgentControls] = {
    "point": POINT_CONTROLS,
}

COMMON_HELP = ["C: camera", "R: reset", "Space: pause", "Esc: quit"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env_name", type=str, default="safe_goal_point",
                   help="Environment name from the CRAX registry")
    p.add_argument("--level", type=int, default=1, choices=[1, 2, 3], help="Difficulty level")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--camera", type=str, default=None,
                   help="Initial camera (default: first camera for the agent)")
    p.add_argument("--width", type=int, default=800, help="Render width in pixels")
    p.add_argument("--height", type=int, default=600, help="Render height in pixels")
    p.add_argument("--fps", type=float, default=None,
                   help="Steps per second (default: real time, i.e. 1 / env.dt)")
    p.add_argument("--max_steps", type=int, default=1000,
                   help="Episode length before an automatic reset")
    return p.parse_args()


def controls_for(env_name: str) -> AgentControls:
    agent = env_name.rsplit("_", 1)[-1]
    if agent not in AGENT_CONTROLS:
        raise ValueError(f"No keyboard controls for agent '{agent}' (env '{env_name}'). "
                         f"Supported agents: {sorted(AGENT_CONTROLS)}")
    return AGENT_CONTROLS[agent]


class Viewer:
    """Renders MJX pipeline states with MuJoCo and blits them into a pygame window."""

    def __init__(self, mj_model: mujoco.MjModel, width: int, height: int):
        self._model = mj_model
        self._data = mujoco.MjData(mj_model)
        # The default offscreen framebuffer is 640x480; grow it to fit the window.
        mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, width)
        mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, height)
        self._renderer = mujoco.Renderer(mj_model, height=height, width=width)
        self.screen = pygame.display.set_mode((width, height))

    def draw(self, pipeline_state, camera: str) -> None:
        d = self._data
        d.qpos[:] = np.asarray(pipeline_state.qpos)
        d.qvel[:] = np.asarray(pipeline_state.qvel)
        if self._model.nmocap > 0:
            d.mocap_pos[:] = np.asarray(pipeline_state.mocap_pos)
            d.mocap_quat[:] = np.asarray(pipeline_state.mocap_quat)
        mujoco.mj_forward(self._model, d)
        self._renderer.update_scene(d, camera=camera)
        frame = self._renderer.render()
        # pygame surfaces are (width, height, 3); MuJoCo frames are (height, width, 3).
        pygame.surfarray.blit_array(self.screen, frame.swapaxes(0, 1))

    def close(self) -> None:
        self._renderer.close()


def draw_hud(screen, font, lines: List[tuple]) -> None:
    y = 8
    for text, color in lines:
        shadow = font.render(text, True, (0, 0, 0))
        screen.blit(shadow, (11, y + 1))
        screen.blit(font.render(text, True, color), (10, y))
        y += font.get_linesize()


def main() -> None:
    args = parse_args()
    controls = controls_for(args.env_name)

    env = envs.get_environment(args.env_name, level=args.level)
    mj_model = env.sys.mj_model
    available = {mj_model.camera(i).name for i in range(mj_model.ncam)}
    cameras = [c for c in controls.cameras if c in available] or [-1]
    cam_idx = cameras.index(args.camera) if args.camera in cameras else 0

    print(f"env={args.env_name} level={args.level} action_size={env.action_size}")
    print("Compiling reset/step (first frame may take a while)...")
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    pygame.init()
    pygame.display.set_caption(f"CRAX - {args.env_name} (level {args.level})")
    viewer = Viewer(mj_model, args.width, args.height)
    font = pygame.font.SysFont("dejavusansmono", 16, bold=True)
    clock = pygame.time.Clock()
    fps = args.fps if args.fps is not None else 1.0 / float(env.dt)

    rng = jax.random.PRNGKey(args.seed)

    def new_episode(rng):
        rng, reset_rng = jax.random.split(rng)
        return rng, reset_fn(reset_rng)

    rng, state = new_episode(rng)
    episode, steps, ep_return, ep_cost = 1, 0, 0.0, 0.0
    paused, running = False, True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_c:
                        cam_idx = (cam_idx + 1) % len(cameras)
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        rng, state = new_episode(rng)
                        episode, steps, ep_return, ep_cost = episode + 1, 0, 0.0, 0.0

            action = controls.action_fn(pygame.key.get_pressed())
            if not paused:
                state = step_fn(state, action)
                steps += 1
                ep_return += float(state.reward)
                ep_cost += float(state.info.get("cost", state.metrics.get("cost", 0.0)))
                if bool(state.done) or steps >= args.max_steps:
                    print(f"episode {episode}: steps={steps} return={ep_return:.2f} cost={ep_cost:.2f}")
                    rng, state = new_episode(rng)
                    episode, steps, ep_return, ep_cost = episode + 1, 0, 0.0, 0.0

            viewer.draw(state.pipeline_state, cameras[cam_idx])
            step_cost = float(state.metrics.get("cost", 0.0))
            hud = [
                (f"Episode {episode}  step {steps}/{args.max_steps}", (240, 240, 240)),
                (f"Return {ep_return:8.2f}", (80, 230, 80)),
                (f"Cost   {ep_cost:8.2f}", (255, 90, 90) if step_cost > 0 else (230, 150, 150)),
                (f"Action [{', '.join(f'{a:+.0f}' for a in action)}]", (200, 200, 255)),
                (f"Camera {cameras[cam_idx]}" + ("   PAUSED" if paused else ""), (240, 240, 120)),
                ("  ".join(controls.help_lines + COMMON_HELP), (200, 200, 200)),
            ]
            draw_hud(viewer.screen, font, hud)
            pygame.display.flip()
            clock.tick(fps)
    finally:
        viewer.close()
        pygame.quit()


if __name__ == "__main__":
    main()
