"""Download the early-fusion comparison runs from WandB.

The fusion study trains the same (env, level, algo) with different ways of
combining pixel and proprioceptive observations:

    vector  -> state only            (no --vision)
    pixel   -> pixels only           (--vision --vision_obs_mode pixels)
    concat  -> pixels + state concat (--vision_obs_mode pixels+state --vision_early_fusion concat)
    sum     -> pixels + state sum    (... --vision_early_fusion sum)
    film    -> pixels + state FiLM   (... --vision_early_fusion film)

The baseline `main_results.obs_mode_segment` keys the folder on the *camera*
only, so concat/sum/film (same env, level, algo and camera) would overwrite each
other. This script instead lays runs out by fusion mode:

    vector -> data/<env>/level_<l>/<algo>/seed_<n>.parquet
    others -> data/<env>/level_<l>/<algo>/fusion_<mode>/seed_<n>.parquet

so `plotting_results.fusion_comparison` can read one folder per mode.

    python -m results.download.fusion_comparison --project my-crax-project --levels 1 2
"""

import argparse
import os
from typing import Optional, Set

import wandb
from wandb.apis.public import Run

from results import cli
from results.common import (
    fusion_mode_dir,
    fusion_mode_of_config,
    get_metrics_for_env,
    results_path,
)
from results.download.main_results import build_filters


def store_fusion_run(run: Run, args: argparse.Namespace,
                     written: Optional[Set] = None) -> None:
    """Write one run's history to its per-(mode, seed) parquet, keyed by fusion mode."""
    config = run.config
    mode = fusion_mode_of_config(config)
    if mode is None or mode not in args.fusion_modes:
        return

    seed = config["seed"]
    env = config["env_name"]
    level = config["difficulty"]
    algo = config["alg"]

    metrics = get_metrics_for_env(env, args.metrics)

    folder_path = results_path(
        args.output, env, f"level_{level}", algo, fusion_mode_dir(mode)
    )
    os.makedirs(folder_path, exist_ok=True)
    file_path = folder_path / f"seed_{seed}.parquet"

    # Newest-first iteration: an already-written cell is the latest run for it.
    if written is not None and file_path in written:
        print(f"Skipping older duplicate of {file_path.name}: run {run.id}")
        return
    if file_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {file_path}")
        return

    try:
        df = run.history(keys=metrics)
        if df is None or df.empty:
            print(f"No history for run {run.id}")
            return
        df.to_parquet(file_path)
        if written is not None:
            written.add(file_path)
        print(f"Saved {mode} run {run.id} to {file_path}")
    except Exception as e:
        print(f"Error downloading data for run: {run.id}: {e}")


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # build_filters selects by env/level/algo/seed/state/tags; the fusion mode is
    # decided per run from its config, so we pull every obs modality here.
    filters = build_filters(args)
    runs = api.runs(args.project, filters=filters, order="-created_at", per_page=200)

    written: Set = set()
    for run in runs:
        store_fusion_run(run, args, written)


def build_args() -> argparse.ArgumentParser:
    p = cli.download_parser(
        "Download the early-fusion (vision/proprioception) comparison runs.",
        # The fusion study only has runs for the Point tasks under CRPO.
        envs=["safe_goal_point", "safe_push_point", "safe_circle_point"],
        algos=["crpo"],
        seeds=list(range(0, 5)),
        levels=[1, 2],
    )
    p.add_argument("--fusion_modes", type=str, nargs="+", default=list(cli.DEFAULT_FUSION_MODES),
                   choices=cli.DEFAULT_FUSION_MODES,
                   help="Fusion modes to keep (vector, pixel, concat, sum, film).")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())
