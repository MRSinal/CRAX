#!/usr/bin/env python3
"""Trajectory plots of the Push fusion-study policies (Level 2 by default, see --levels).

For every (level, observation / fusion mode) this picks the run of one training
seed from WandB, finds its checkpoint under `models/<run name>`, and calls
`scripts/plot_trajectory.py` on it. Every policy is replayed on the same episode
seed, so within a level all modes face the same hazard layout. A grid with the
levels down the rows and the modes across the columns is written as well.

    vector | pixel | concat | sum | film     (plus mlp / gap with --variants)

Checkpoints are saved on the machine that trained them and are not uploaded to
WandB. List the run directories to copy over with:

    python scripts/plot_push_trajectories.py --list_runs

then copy them into `models/` and run:

    python scripts/plot_push_trajectories.py
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from results.common import (  # noqa: E402
    FUSION_MODES,
    FUSION_VARIANTS,
    TRANSLATIONS,
    fusion_mode_of_config,
    fusion_variant_of_config,
)

VARIANT_SUFFIX = {"base": "", "mlp": " + MLP", "gap": " + GAP"}

Cell = Tuple[int, str, str]  # (level, mode, variant)


def find_runs(args: argparse.Namespace) -> Dict[Cell, str]:
    """Newest finished run name per (level, mode, variant) for the chosen training seed."""
    import wandb

    filters = {"config.env_name": args.env, "config.alg": args.algo,
               "config.difficulty": {"$in": args.levels}, "config.seed": args.run_seed,
               "state": "finished"}
    runs: Dict[Cell, str] = {}
    for run in wandb.Api().runs(args.project, filters=filters, order="-created_at", per_page=200):
        variant = fusion_variant_of_config(run.config)
        if variant not in args.variants:
            continue
        mode = fusion_mode_of_config(run.config, variant=variant)
        if mode not in args.modes:
            continue
        runs.setdefault((int(run.config["difficulty"]), mode, variant), run.name)
    return runs


def cells(args: argparse.Namespace) -> List[Cell]:
    """Grid cells in plotting order; vector/pixel only exist as the base variant."""
    out = []
    for level in args.levels:
        for variant in args.variants:
            for mode in args.modes:
                if variant != "base" and mode in ("vector", "pixel"):
                    continue
                out.append((level, mode, variant))
    return out


def label(mode: str, variant: str) -> str:
    return TRANSLATIONS.get(mode, mode) + VARIANT_SUFFIX[variant]


def plot_cell(run_name: str, out_path: Path, args: argparse.Namespace) -> bool:
    cmd = [sys.executable, str(ROOT_DIR / "scripts" / "plot_trajectory.py"),
           "--checkpoint", run_name, "--seed", str(args.episode_seed),
           "--color_by", args.color_by, "--wandb_project", args.project,
           "--output", str(out_path)]
    if args.extent is not None:
        cmd += ["--extent", str(args.extent)]
    if args.stochastic:
        cmd.append("--stochastic")
    print(f"\n$ {' '.join(cmd)}")
    # One process per policy: each JIT-compiles its own env and frees the GPU when done.
    return subprocess.run(cmd, cwd=ROOT_DIR).returncode == 0


def plot_grid(images: Dict[Cell, Path], args: argparse.Namespace) -> Optional[Path]:
    columns = list(dict.fromkeys((m, v) for (_, m, v) in cells(args)))
    if not images:
        return None
    fig, axs = plt.subplots(len(args.levels), len(columns),
                            figsize=(2.6 * len(columns), 2.7 * len(args.levels)), squeeze=False)
    for row, level in enumerate(args.levels):
        for col, (mode, variant) in enumerate(columns):
            ax = axs[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            img = images.get((level, mode, variant))
            if img is not None:
                ax.imshow(plt.imread(img))
            else:
                ax.text(0.5, 0.5, "No checkpoint", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            if row == 0:
                ax.set_title(label(mode, variant), fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Level {level}", fontsize=11)
    fig.tight_layout()
    out_path = args.output_dir / f"{args.env}_trajectories_grid_{args.color_by}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved grid: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="crax", help="WandB project")
    p.add_argument("--env", default="safe_push_point")
    p.add_argument("--algo", default="crpo")
    p.add_argument("--levels", type=int, nargs="+", default=[2])
    p.add_argument("--modes", nargs="+", default=list(FUSION_MODES), choices=FUSION_MODES)
    p.add_argument("--variants", nargs="+", default=["base"], choices=FUSION_VARIANTS,
                   help="Add mlp / gap for the proprio-MLP and GAP runs of concat/sum/film.")
    p.add_argument("--run_seed", type=int, default=0, help="Training seed of the policy to replay")
    p.add_argument("--episode_seed", type=int, default=0,
                   help="Episode seed (hazard layout), shared by every policy of a level")
    p.add_argument("--color_by", choices=["time", "cost"], default="cost")
    p.add_argument("--extent", type=float, default=None,
                   help="Fixed half-size of the rendered square, so all panels share the framing")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--output_dir", type=Path, default=ROOT_DIR / "visualizations" / "push_trajectories")
    p.add_argument("--list_runs", action="store_true",
                   help="Only print the run directories needed under models/, then exit")
    p.add_argument("--skip_existing", action="store_true", help="Reuse panels that were already plotted")
    args = p.parse_args()

    runs = find_runs(args)
    todo = cells(args)
    missing_runs = [c for c in todo if c not in runs]
    for level, mode, variant in missing_runs:
        print(f"[warning] No finished WandB run for Level {level} {label(mode, variant)} seed {args.run_seed}")

    needed = [runs[c] for c in todo if c in runs]
    if args.list_runs:
        print("\nRun directories needed under models/:")
        print("\n".join(needed))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images: Dict[Cell, Path] = {}
    no_ckpt, failed = [], []
    for cell in todo:
        if cell not in runs:
            continue
        level, mode, variant = cell
        run_name = runs[cell]
        if not (ROOT_DIR / "models" / run_name).is_dir():
            no_ckpt.append(run_name)
            continue
        out_path = args.output_dir / (f"{args.env}_L{level}_{mode}"
                                      f"{'' if variant == 'base' else '_' + variant}"
                                      f"_ep{args.episode_seed}_{args.color_by}.png")
        if (args.skip_existing and out_path.exists()) or plot_cell(run_name, out_path, args):
            images[cell] = out_path
        else:
            failed.append(run_name)

    plot_grid(images, args)
    print(f"\nPlotted {len(images)}/{len(todo)} panels.")
    if no_ckpt:
        print(f"{len(no_ckpt)} checkpoint(s) not found under models/ (copy them from the training machine):")
        print("\n".join(f"  {r}" for r in no_ckpt))
    if failed:
        print(f"{len(failed)} plot(s) failed:")
        print("\n".join(f"  {r}" for r in failed))


if __name__ == "__main__":
    main()
