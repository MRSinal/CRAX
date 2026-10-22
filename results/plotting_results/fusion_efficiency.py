"""Training efficiency of the early-fusion modes (appendix figure).

Three bar panels, one bar per fusion mode (vector | pixel | concat | sum | film):

    Throughput (steps/s) | GPU utilisation (%) | Wall-clock time (min)

Each bar is the mean over seeds. The numbers come straight from WandB rather than
the downloaded parquet files: throughput is the mean of the logged `training/sps`,
GPU utilisation is the mean of the `system/gpu.0.gpu` stream, and wall-clock time
is the run's `_runtime`. Only runs on `--gpu` are kept so that the modes are
compared on the same hardware (Push Level 1 mixes A40 and A100, Level 2 is A100 only).

The per-run numbers are cached to `<input>/efficiency/<env>_level_<l>_<algo>.csv`;
pass `--refresh` to pull them again, e.g. after more seeds finish.

    python -m results.plotting_results.fusion_efficiency --project crax \
        --env safe_push_point --level 2 --algo crpo
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results import cli
from results.common import (
    FUSION_MODE_COLORS,
    TRANSLATIONS,
    fusion_mode_of_config,
    results_path,
    set_mpl_style,
)

GPU_UTIL_SUFFIX = "gpu.0.gpu"


def _mean_sps(run) -> Optional[float]:
    hist = run.history(keys=["training/sps"], samples=2000)
    if hist is None or hist.empty or "training/sps" not in hist:
        return run.summary.get("training/sps")
    return float(hist["training/sps"].dropna().mean())


def _mean_gpu_util(run) -> Optional[float]:
    sysdf = run.history(stream="system", samples=2000)
    cols = [c for c in sysdf.columns if c.endswith(GPU_UTIL_SUFFIX)]
    if not cols:
        return None
    return float(sysdf[cols[0]].dropna().mean())


def fetch_runs(args: argparse.Namespace) -> pd.DataFrame:
    """One row per finished (mode, seed) run of env/level/algo, newest run per cell."""
    import wandb

    api = wandb.Api()
    filters = {"config.env_name": args.env, "config.alg": args.algo,
               "config.difficulty": args.level, "state": "finished"}
    rows, seen = [], set()
    for run in api.runs(args.project, filters=filters, order="-created_at", per_page=200):
        mode = fusion_mode_of_config(run.config)  # base runs only, no MLP/GAP
        seed = run.config.get("seed")
        if mode is None or mode not in args.fusion_modes or (mode, seed) in seen:
            continue
        seen.add((mode, seed))
        rows.append(dict(
            run_id=run.id, mode=mode, seed=seed,
            gpu=(run.metadata or {}).get("gpu"),
            num_timesteps=run.config.get("num_timesteps"),
            runtime_s=run.summary.get("_runtime"),
            sps=_mean_sps(run),
            gpu_util=_mean_gpu_util(run),
        ))
        print(f"Fetched {mode} seed {seed}: {run.id}")
    return pd.DataFrame(rows)


def load_runs(args: argparse.Namespace) -> pd.DataFrame:
    cache = results_path(args.input, "efficiency",
                         f"{args.env}_level_{args.level}_{args.algo}.csv")
    if cache.exists() and not args.refresh:
        print(f"Using cached efficiency data: {cache}")
        return pd.read_csv(cache)
    df = fetch_runs(args)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"Cached efficiency data: {cache}")
    return df


def _fmt_k(v: float, _pos=None) -> str:
    return f"{v / 1000:.0f}K" if v >= 1000 else f"{v:.0f}"


PANELS = [
    # (column, title, value label)
    ("sps", "Throughput (steps/s)", _fmt_k),
    ("gpu_util", "GPU utilisation (%)", lambda v: f"{v:.0f}%"),
    ("wall_min", "Wall-clock time (min)", lambda v: f"{v:.0f}"),
]


def plot(df: pd.DataFrame, args: argparse.Namespace) -> Path:
    set_mpl_style()
    modes: List[str] = [m for m in args.fusion_modes if m in set(df["mode"])]
    df = df.assign(wall_min=df["runtime_s"] / 60.0)
    agg = df.groupby("mode")[[c for c, _, _ in PANELS]].mean().reindex(modes)

    fig, axs = plt.subplots(1, len(PANELS), figsize=(args.panel_w * len(PANELS), args.panel_h))
    x = np.arange(len(modes))
    labels = [TRANSLATIONS.get(m, m) for m in modes]
    colors = [FUSION_MODE_COLORS.get(m) for m in modes]

    for ax, (col, title, fmt) in zip(axs, PANELS):
        vals = agg[col].to_numpy(dtype=float)
        bars = ax.bar(x, vals, color=colors, width=0.7)
        top = np.nanmax(vals)
        ax.set_ylim(0, (100 * 1.18) if col == "gpu_util" else top * 1.18)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.02, fmt(v),
                        ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        if col == "sps":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_k))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out_dir = results_path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_{args.env}_level_{args.level}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return out_path


def main(args: argparse.Namespace) -> None:
    df = load_runs(args)
    if df.empty:
        raise SystemExit("No finished runs found on WandB for this env/level/algo.")
    if args.gpu:
        dropped = df[~df["gpu"].fillna("").str.contains(args.gpu)]
        if not dropped.empty:
            print(f"Dropping {len(dropped)} run(s) not on {args.gpu}: "
                  + ", ".join(f"{r.mode}/seed{r.seed} ({r.gpu})" for r in dropped.itertuples()))
        df = df[df["gpu"].fillna("").str.contains(args.gpu)]
    summary = df.groupby("mode").agg(n=("seed", "size"), gpu=("gpu", "first"),
                                     sps=("sps", "mean"), gpu_util=("gpu_util", "mean"),
                                     wall_min=("runtime_s", lambda s: s.mean() / 60))
    print(summary.reindex([m for m in args.fusion_modes if m in summary.index]).to_string())
    plot(df, args)


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[cli.figure_args(out_name="fusion_efficiency",
                                 omit=("grid", "max_cols", "smoothing_window"))],
    )
    p.set_defaults(panel_w=3.0, panel_h=3.0)
    p.add_argument("--project", type=str, default="crax", help="WandB project")
    p.add_argument("--input", type=str, default=cli.DEFAULT_DATA_DIR,
                   help="Base data directory (the cache goes under <input>/efficiency)")
    p.add_argument("--env", type=str, default="safe_push_point")
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--algo", type=str, default="crpo")
    p.add_argument("--fusion_modes", type=str, nargs="+", default=list(cli.DEFAULT_FUSION_MODES),
                   choices=cli.DEFAULT_FUSION_MODES)
    p.add_argument("--gpu", type=str, default="A100",
                   help="Keep only runs whose GPU name contains this ('' keeps all).")
    p.add_argument("--refresh", action="store_true", help="Re-pull from WandB, ignoring the cache.")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())
