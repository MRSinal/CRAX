"""Compare early-fusion techniques for combining pixels and proprioception.

One figure per environment. Each figure is a grid with the metrics down the rows
(Reward on top, Cost below) and the difficulty levels across the columns, so
Level 1 and Level 2 sit next to each other. Every panel draws one final-performance
bar per fusion mode:

    vector | pixel | concat | sum | film

with the safety threshold drawn on the cost panels.

Data layout (see `download.fusion_comparison`):
    vector -> data/<env>/level_<l>/<algo>/seed_<n>.parquet
    others -> data/<env>/level_<l>/<algo>/fusion_<mode>/seed_<n>.parquet

    # the fusion study: Point tasks under CRPO, levels 1 and 2
    python -m results.plotting_results.fusion_comparison \
        --envs safe_goal_point safe_push_point safe_circle_point \
        --algo crpo --levels 1 2 --output_fig_dir results/figures
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results import cli
from results.common import (
    DEFAULT_METRIC_COLS as METRIC_COLS,
    FUSION_MODE_COLORS,
    TRANSLATIONS,
    align_and_stack,
    fusion_mode_dir,
    get_series,
    legend_ncol,
    results_path,
    set_mpl_style,
)

# (env, level, mode, metric) -> one DataFrame['_step', 'value'] per seed
RunStore = Dict[Tuple[str, int, str, str], List[pd.DataFrame]]


def load_runs(base: Path, env: str, level: int, algo: str, mode: str,
              seeds: List[int], metrics: List[str]) -> RunStore:
    """Load every seed of one (env, level, mode) cell."""
    out: RunStore = {}
    folder = base / env / f"level_{level}" / algo / fusion_mode_dir(mode)
    for metric in metrics:
        key = (env, level, mode, metric)
        out[key] = []
        for seed in seeds:
            fp = folder / f"seed_{seed}.parquet"
            if not fp.exists():
                continue
            df = pd.read_parquet(fp, engine="pyarrow")
            if "_step" not in df:
                continue
            series = get_series(df, algo=algo, metric=metric,
                                metric_cols=METRIC_COLS, env_name=env)
            if series is None:
                continue
            out[key].append(pd.DataFrame({
                "_step": df["_step"].astype(np.int64),
                "value": series.astype(np.float32),
            }).dropna())
    return out


def load_all(args: argparse.Namespace) -> RunStore:
    base = results_path(args.input)
    store: RunStore = {}
    for env in args.envs:
        for level in args.levels:
            for mode in args.fusion_modes:
                store.update(load_runs(base, env, level, args.algo, mode,
                                       args.seeds, args.metrics))
    return store


def _final_value(runs: List[pd.DataFrame], last_k: int) -> Optional[Tuple[float, float]]:
    """(mean, 95% CI half-width) of the last `last_k` points, averaged over seeds."""
    if not runs:
        return None
    _, vals = align_and_stack(runs)
    if vals.size == 0:
        return None
    k = min(last_k, vals.shape[1])
    per_seed = vals[:, -k:].mean(axis=1)
    ci = 1.96 * per_seed.std() / np.sqrt(max(per_seed.size, 1))
    return float(per_seed.mean()), float(ci)


def _clipped_yerr(heights: List[float], errors: List[float]) -> np.ndarray:
    """Asymmetric yerr whose lower whisker stops at zero (reward/cost are >= 0)."""
    lower = [min(err, height) if height >= 0 else err
             for height, err in zip(heights, errors)]
    return np.array([lower, errors])


def plot_env(store: RunStore, args: argparse.Namespace, env: str) -> Optional[Path]:
    """One figure for `env`: metrics down the rows, levels across the columns."""
    set_mpl_style()
    metrics, levels, modes = args.metrics, args.levels, args.fusion_modes

    n_rows, n_cols = len(metrics), len(levels)
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(args.panel_w * n_cols, args.panel_h * n_rows),
                            squeeze=False)
    fig.suptitle(TRANSLATIONS.get(env, env))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.18,
                        wspace=0.25, hspace=0.30)

    handles: Dict[str, plt.Artist] = {}
    drew_any = False

    for metric_i, metric in enumerate(metrics):
        for level_i, level in enumerate(levels):
            ax = axs[metric_i][level_i]

            positions, heights, errors, colors, drawn = [], [], [], [], []
            for i, mode in enumerate(modes):
                stat = _final_value(store.get((env, level, mode, metric), []), args.last_k)
                if stat is None:
                    continue
                mean, ci = stat
                positions.append(i)
                heights.append(mean)
                errors.append(ci)
                colors.append(FUSION_MODE_COLORS.get(mode))
                drawn.append(mode)

            if positions:
                drew_any = True
                bars = ax.bar(positions, heights, yerr=_clipped_yerr(heights, errors),
                              capsize=4, color=colors, edgecolor="black", linewidth=0.6)
                for mode, bar in zip(drawn, bars):
                    handles.setdefault(mode, bar)
                if min(heights) >= 0.0:
                    ax.set_ylim(bottom=0.0)

            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels([""] * len(modes))  # modes labelled by the legend
            ax.axhline(0.0, color="black", linewidth=0.8)

            if metric_i == 0:
                ax.set_title(f"Level {level}", pad=8)
            if level_i == 0:
                ax.set_ylabel(TRANSLATIONS.get(metric, metric.capitalize()))
            if metric == "cost" and not args.no_threshold:
                thr = ax.axhline(args.threshold, linestyle="--", color="red", linewidth=1.8)
                handles.setdefault("Threshold", thr)
            if args.grid:
                ax.grid(True, axis="y", linestyle="--", linewidth=0.9, alpha=0.45)

    if not drew_any:
        plt.close(fig)
        print(f"No data for env '{env}', skipping.")
        return None

    labels, hs = zip(*handles.items())
    labels = [TRANSLATIONS.get(lbl, lbl) for lbl in labels]
    fig.legend(hs, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=legend_ncol(len(modes), len(labels)), fancybox=True, shadow=True)

    out_dir = results_path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_{env}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return out_path


def main(args: argparse.Namespace) -> None:
    store = load_all(args)
    if not any(store.values()):
        raise SystemExit(
            "No data loaded. Check --input/--envs/--algo/--levels/--seeds, and that the "
            "fusion runs were downloaded (python -m results.download.fusion_comparison)."
        )
    for env in args.envs:
        plot_env(store, args, env)


def build_args() -> argparse.ArgumentParser:
    p = cli.plot_parser(
        "Compare early-fusion techniques (vector/pixel/concat/sum/FiLM) per env, "
        "with levels side by side.",
        level_arg="list",          # Level 1 and Level 2 next to each other
        stats=True,
        omit=("algos", "ci_method", "last_frac", "max_cols"),
        out_name="fusion_comparison",
        envs=["safe_goal_point", "safe_push_point", "safe_circle_point"],
        seeds=list(range(0, 5)),
        levels=[1, 2],
        panel_w=3.1,
        panel_h=2.3,
    )
    p.add_argument("--algo", type=str, default="crpo",
                   help="Algorithm whose fusion runs to plot (the study used crpo).")
    p.add_argument("--fusion_modes", type=str, nargs="+", default=list(cli.DEFAULT_FUSION_MODES),
                   choices=cli.DEFAULT_FUSION_MODES,
                   help="Fusion modes to show, in bar order.")
    p.add_argument("--last_k", type=int, default=10,
                   help="Logged points averaged to get each run's final value.")
    p.add_argument("--no_threshold", action="store_true", help="Hide safety threshold lines.")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())
