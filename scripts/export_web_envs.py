#!/usr/bin/env python3
"""Export CRAX point-agent environments for the in-browser player.

The project page (``docs/``) runs the environments with the MuJoCo WebAssembly
bindings instead of MJX. For every (env, level) pair this script writes:

  * ``<name>.mjb``  - the compiled MuJoCo model, exactly as the env builds it
                      (solver, timestep and damping overrides included).
  * ``<name>.json`` - the task parameters the JavaScript port of
                      ``reset``/``step`` needs (hazards, goals, rewards, costs).

The ``.mjb`` format is tied to the MuJoCo version, so the page must load the
same MuJoCo release as the one used here (see ``MUJOCO_VERSION`` in
``docs/play/crax-env.js``).

Usage:
    python scripts/export_web_envs.py [--out docs/play/envs]
"""
import argparse
import json
import os

import mujoco
import numpy as np

from crax import envs

ENV_NAMES = ["safe_goal_point", "safe_push_point", "safe_circle_point", "safe_button_point"]
LEVELS = [1, 2, 3]

# The agent XMLs reserve njmax=3000 / nconmax=1000, which CPU MuJoCo turns into
# a ~470 MB arena per MjData. That exceeds what a WebAssembly heap can hold
# once a few envs are cached, so the exported models use a small fixed arena.
WEB_ARENA_BYTES = 8 * 1024 * 1024
_from_xml_path = mujoco.MjModel.from_xml_path


def _from_xml_path_small_arena(path, *args, **kwargs):
    if args or kwargs:
        return _from_xml_path(path, *args, **kwargs)
    spec = mujoco.MjSpec.from_file(path)
    spec.memory, spec.njmax, spec.nconmax = WEB_ARENA_BYTES, -1, -1
    return spec.compile()


def _f(x):
    return float(np.asarray(x))


def _list(x):
    return np.asarray(x, dtype=np.float64).tolist()


def hazard_spec(h, mocap_id: int) -> dict:
    shape, param = h.get_keepout_shape()
    return {
        "type": h.hazard_type,
        "geom_id": int(h.geom_id),
        "mocap_id": int(mocap_id),
        "size": _list(getattr(h, "size_xy", None) or [h.size]),
        "collidable": bool(h.collidable),
        "fixed": bool(h.fixed),
        "travel": _f(getattr(h, "travel", 0.0)),
        "keepout": _f(h.get_keepout_radius()),
        "keepout_shape": shape,
        "keepout_param": _list(param),
    }


def common_spec(env, env_name: str, level: int) -> dict:
    m = env.sys.mj_model
    hazards = env._hazard_manager.hazards
    return {
        "env_name": env_name,
        "task": env_name.split("_")[1],
        "level": level,
        "mujoco_version": mujoco.__version__,
        "n_frames": int(env._n_frames),
        "dt": _f(env.dt),
        "episode_length": int(getattr(env, "episode_length", 1000)),
        "agent_body": int(env._agent_body),
        "agent_geom_ids": [int(i) for i in np.asarray(env._agent_geom_ids)],
        "healthy_z_range": _list(env._healthy_z_range),
        "reset_noise_scale": _f(env._reset_noise_scale),
        "placement_extents": _list(env._placement_extents),
        "placement_margin": _f(env._placement_margin),
        "max_placement_attempts": int(env._max_placement_attempts),
        "agent_keepout": _f(env._agent_keepout),
        "proximity_cost_scaler": _f(env._proximity_cost_scaler),
        "collision_cost": _f(env._collision_cost),
        "num_movable_hazards": int(env._num_movable_hazards),
        "hazards": [hazard_spec(h, mid) for h, mid in zip(hazards, env._hazard_mocap_ids)],
        "cameras": [m.camera(i).name for i in range(m.ncam)],
    }


def hazard_shapes(env) -> dict:
    return {
        "hazard_is_rect": [bool(b) for b in np.asarray(env._hazard_is_rect)],
        "hazard_half_extents": _list(env._hazard_half_extents),
        "hazard_radii": _list(env._hazard_radii),
    }


def goal_spec(env) -> dict:
    return {
        "goal_mocap_ids": [int(i) for i in env._goal_mocap_ids],
        "goal_type_ids": [int(i) for i in np.asarray(env._goal_type_ids)],
        "goal_radii": _list(env._goal_radii),
        "goal_box_he": _list(env._goal_box_he),
        "goal_yaws": _list(env._goal_yaws),
        "goal_keepouts": _list(env._goal_keepouts),
        **hazard_shapes(env),
        "reward_goal": _f(env._reward_goal),
        "reward_distance": _f(env._reward_distance),
    }


def task_spec(env, task: str) -> dict:
    if task == "goal":
        return goal_spec(env)
    if task == "push":
        spec = goal_spec(env)
        spec.update({
            "block_body": int(env._block_body),
            "goal_velocity": _f(env._goal_velocity),
            "reward_agent_block": _f(env._reward_agent_block),
        })
        return spec
    if task == "circle":
        return {
            "goal_mocap_ids": [int(i) for i in env._goal_mocap_ids],
            "goal_positions": _list(env._goal_positions),
            "hazard_placement_extents": _list(env._hazard_placement_extents),
            "circle_radius": _f(env._circle_radius),
            "circle_center": _list(env._circle_center),
            "reward_factor": _f(env._reward_factor),
            "boundary_x": None if env._boundary_x is None else _f(env._boundary_x),
            "boundary_y": None if env._boundary_y is None else _f(env._boundary_y),
            "boundary_cost": _f(env._boundary_cost),
            "init_xy_range": _f(env._init_xy_range),
            "init_angle_range": _f(env._init_angle_range),
        }
    if task == "button":
        return {
            **hazard_shapes(env),
            "button_mocap_ids": [int(i) for i in env._button_mocap_ids],
            "button_geom_ids": [int(i) for i in np.asarray(env._button_geom_ids)],
            "goal_marker_mocap_id": int(env._goal_marker_mocap_id),
            "button_height": _f(env._button_height),
            "button_keepout": _f(env._button_keepout),
            "marker_z": _f(env._marker_z),
            "buttons_constrained": bool(env._buttons_constrained),
            "resampling_delay": int(env._resampling_delay),
            "continue_goal": bool(env._continue_goal),
            "max_layout_attempts": int(env._max_layout_attempts),
            "reward_goal": _f(env._reward_goal),
            "reward_distance": _f(env._reward_distance),
            "wrong_button_cost": _f(env._wrong_button_cost),
        }
    raise ValueError(f"Unsupported task '{task}'")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join("docs", "play", "envs"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mujoco.MjModel.from_xml_path = staticmethod(_from_xml_path_small_arena)

    index = []
    for env_name in ENV_NAMES:
        for level in LEVELS:
            env = envs.get_environment(env_name, level=level)
            spec = common_spec(env, env_name, level)
            spec.update(task_spec(env, spec["task"]))

            name = f"{env_name}_level{level}"
            mujoco.mj_saveModel(env.sys.mj_model, os.path.join(args.out, f"{name}.mjb"), None)
            with open(os.path.join(args.out, f"{name}.json"), "w") as f:
                json.dump(spec, f, indent=1)
            index.append({"env_name": env_name, "level": level, "file": name})
            print(f"wrote {name} (nbody={env.sys.mj_model.nbody}, hazards={len(spec['hazards'])})")

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
