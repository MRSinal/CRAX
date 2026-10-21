"""Compare GAP enabled vs. GAP disabled on the same fusion architecture.

Both variants add an MLP over the proprioception in the policy head
(`--proprio_mlp`); they differ only in `--gap_lambda`:

    mlp -> GAP disabled (gap_lambda 0, wandb group *_mlp)
    gap -> GAP enabled  (gap_lambda > 0, wandb group *_gap)

so for every fusion mode (concat | sum | film) the two runs share the architecture.
Only Level 2 has GAP runs, so that is the default (`--levels` overrides it).
Pass `--variants base mlp gap` to also show the original runs without the MLP.

Two figures are written:

  * `<out_name>_curves.pdf`: one row per fusion mode, and in each row a separate
    Reward | Cost pair per level, one line per variant (mean over seeds, 95% CI).
  * `<out_name>_bars.pdf`: final performance, metrics down the rows and levels
    across the columns, bars grouped by fusion mode with one bar per variant.

A table of the final values and the GAP - no-GAP difference is printed as well.

Data layout (see `download.fusion_comparison`):
    data/<env>/level_<l>/<algo>/fusion_<mode>_<variant>/seed_<n>.parquet

    python -m results.download.fusion_comparison --project crax \
        --envs safe_push_point --variants mlp gap
    python -m results.plotting_results.gap_comparison
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results import cli
from results.common import (
    FUSION_VARIANTS,
    FUSION_VARIANT_COLORS,
    TRANSLATIONS,
    align_and_stack,
    legend_ncol,
    moving_average,
    results_path,
    set_mpl_style,
)
from results.plotting_results.fusion_comparison import _clipped_yerr, _final_value, load_runs

PIXEL_STATE_MODES = ["concat", "sum", "film"]
VARIANT_LABELS = {
    "base": "No MLP",
    "mlp": "GAP disabled",
    "gap": "GAP enabled",
}

# (level, mode, variant, metric) -> one DataFrame['_step', 'value'] per seed
RunStore = Dict[Tuple[int, str, str, str], List[pd.DataFrame]]


def load_all(args: argparse.Namespace) -> RunStore:
    base = results_path(args.input)
    store: RunStore = {}
    for level in args.levels:
        for mode in args.fusion_modes:
            for variant in args.variants:
                runs = load_runs(base, args.env, level, args.algo, mode,
                                 args.seeds, args.metrics, variant=variant)
                for (_, _, _, metric), dfs in runs.items():
                    store[(level, mode, variant, metric)] = dfs
    return store


def _save(fig, args: argparse.Namespace, kind: str) -> Path:
    out_dir = results_path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_{kind}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return out_path


def _legend(fig, handles: Dict[str, plt.Artist], n_variants: int) -> None:
    labels, hs = zip(*handles.items())
    labels = [VARIANT_LABELS.get(lbl, TRANSLATIONS.get(lbl, lbl)) for lbl in labels]
    fig.legend(hs, labels, loc="outside lower center",
               ncol=legend_ncol(n_variants, len(labels)), fancybox=True, shadow=True)


def plot_curves(store: RunStore, args: argparse.Namespace) -> Optional[Path]:
    """Rows are fusion modes; each row holds a Reward | Cost pair per level."""
    set_mpl_style()
    modes, levels, metrics, variants = args.fusion_modes, args.levels, args.metrics, args.variants

    n_cols = len(levels) * len(metrics)
    fig = plt.figure(figsize=(args.panel_w * n_cols, args.panel_h * len(modes) + 0.6),
                     layout="constrained")
    fig.suptitle(TRANSLATIONS.get(args.env, args.env))
    subfigs = np.asarray(fig.subfigures(len(modes), len(levels), wspace=0.04, hspace=0.04),
                         dtype=object).reshape(len(modes), len(levels))

    handles: Dict[str, plt.Artist] = {}
    drew_any = False

    for mode_i, mode in enumerate(modes):
        for level_i, level in enumerate(levels):
            sub = subfigs[mode_i][level_i]
            sub.suptitle(f"{TRANSLATIONS.get(mode, mode)} - Level {level}")
            axs = np.atleast_1d(sub.subplots(1, len(metrics)))

            for metric_i, metric in enumerate(metrics):
                ax = axs[metric_i]
                non_negative = True

                for variant in variants:
                    runs = store.get((level, mode, variant, metric), [])
                    if not runs:
                        continue
                    steps, vals = align_and_stack(runs)
                    if vals.size == 0:
                        continue
                    drew_any = True
                    variant_non_negative = float(vals.min()) >= 0.0
                    non_negative &= variant_non_negative

                    x = steps.astype(float)
                    mean = vals.mean(axis=0)
                    ci = 1.96 * vals.std(axis=0) / np.sqrt(max(vals.shape[0], 1))
                    if args.smoothing_window:
                        mean = moving_average(mean, args.smoothing_window)
                        ci = moving_average(ci, args.smoothing_window)

                    lower = mean - ci
                    if variant_non_negative:
                        lower = np.maximum(lower, 0.0)

                    line, = ax.plot(x, mean, color=FUSION_VARIANT_COLORS.get(variant))
                    ax.fill_between(x, lower, mean + ci, alpha=0.25, color=line.get_color())
                    handles.setdefault(variant, line)

                if not ax.lines:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
                    ax.set_xticks([])
                    ax.set_yticks([])
                if non_negative:
                    ax.set_ylim(bottom=0.0)
                if args.x_max is not None:
                    ax.set_xlim(0.0, args.x_max)
                else:
                    ax.set_xlim(left=0.0)
                if ax.get_ylim()[1] >= 1000:
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                ax.yaxis.get_major_formatter().set_useOffset(False)

                ax.set_xlabel("Steps")
                ax.set_ylabel(TRANSLATIONS.get(metric, metric.capitalize()))
                if metric == "cost" and not args.no_threshold:
                    thr = ax.axhline(args.threshold, linestyle="--", color="red", linewidth=1.8)
                    handles.setdefault("Threshold", thr)
                if args.grid:
                    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.45)

    if not drew_any:
        plt.close(fig)
        print("No GAP/MLP curve data, skipping learning-curve figure.")
        return None
    _legend(fig, handles, len(variants))
    return _save(fig, args, "curves")


def plot_bars(store: RunStore, args: argparse.Namespace) -> Optional[Path]:
    """Final performance: bars grouped by fusion mode, one bar per variant."""
    set_mpl_style()
    modes, levels, metrics, variants = args.fusion_modes, args.levels, args.metrics, args.variants

    fig, axs = plt.subplots(len(metrics), len(levels),
                            figsize=(max(args.panel_w * 1.3 * len(levels), 5.5), args.panel_h * len(metrics) + 0.6),
                            squeeze=False, layout="constrained")
    fig.suptitle(TRANSLATIONS.get(args.env, args.env))

    width = 0.8 / len(variants)
    handles: Dict[str, plt.Artist] = {}
    drew_any = False

    for metric_i, metric in enumerate(metrics):
        for level_i, level in enumerate(levels):
            ax = axs[metric_i][level_i]
            heights_all = []
            for v_i, variant in enumerate(variants):
                pos, heights, errors = [], [], []
                for m_i, mode in enumerate(modes):
                    stat = _final_value(store.get((level, mode, variant, metric), []), args.last_k)
                    if stat is None:
                        continue
                    pos.append(m_i + (v_i - (len(variants) - 1) / 2) * width)
                    heights.append(stat[0])
                    errors.append(stat[1])
                if not pos:
                    continue
                drew_any = True
                heights_all += heights
                bars = ax.bar(pos, heights, width=width, yerr=_clipped_yerr(heights, errors),
                              capsize=3, color=FUSION_VARIANT_COLORS.get(variant),
                              edgecolor="black", linewidth=0.6)
                handles.setdefault(variant, bars)

            if not heights_all:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_yticks([])
            elif min(heights_all) >= 0.0:
                ax.set_ylim(bottom=0.0)
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels([TRANSLATIONS.get(m, m) for m in modes])
            if metric_i == 0:
                ax.set_title(f"Level {level}")
            if level_i == 0:
                ax.set_ylabel(TRANSLATIONS.get(metric, metric.capitalize()))
            if metric == "cost" and not args.no_threshold:
                thr = ax.axhline(args.threshold, linestyle="--", color="red", linewidth=1.8)
                handles.setdefault("Threshold", thr)
            if args.grid:
                ax.grid(True, axis="y", linestyle="--", linewidth=0.9, alpha=0.45)
                ax.set_axisbelow(True)

    if not drew_any:
        plt.close(fig)
        print("No GAP/MLP data, skipping bar figure.")
        return None
    _legend(fig, handles, len(variants))
    return _save(fig, args, "bars")


def summary_table(store: RunStore, args: argparse.Namespace) -> pd.DataFrame:
    """Final mean ± CI per (level, mode, metric, variant), plus GAP minus no-GAP."""
    rows = []
    for level in args.levels:
        for mode in args.fusion_modes:
            for metric in args.metrics:
                row = dict(level=level, mode=mode, metric=metric)
                for variant in args.variants:
                    runs = store.get((level, mode, variant, metric), [])
                    stat = _final_value(runs, args.last_k)
                    row[variant] = (f"{stat[0]:.1f} ± {stat[1]:.1f} (n={len(runs)})"
                                    if stat else "--")
                    row[f"_{variant}"] = stat[0] if stat else np.nan
                if "mlp" in args.variants and "gap" in args.variants:
                    row["gap - mlp"] = row["_gap"] - row["_mlp"]
                rows.append(row)
    df = pd.DataFrame(rows)
    return df[[c for c in df.columns if not c.startswith("_")]]


def main(args: argparse.Namespace) -> None:
    store = load_all(args)
    if not any(store.values()):
        raise SystemExit(
            "No GAP/MLP data loaded. Download it first:\n"
            "  python -m results.download.fusion_comparison --project crax "
            f"--envs {args.env} --variants {' '.join(args.variants)}"
        )
    plot_curves(store, args)
    plot_bars(store, args)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary_table(store, args).to_string(index=False, float_format="%.1f"))


def build_args() -> argparse.ArgumentParser:
    p = cli.plot_parser(
        "Compare GAP enabled vs. disabled (same proprio-MLP fusion architecture).",
        level_arg="list",
        stats=True,
        omit=("algos", "envs", "ci_method", "last_frac", "max_cols"),
        out_name="gap_comparison",
        seeds=list(range(0, 5)),
        levels=[2],  # the GAP runs only exist for Level 2
        panel_w=3.1,
        panel_h=2.3,
    )
    p.add_argument("--env", type=str, default="safe_push_point",
                   help="Environment (the GAP runs only exist for Push).")
    p.add_argument("--algo", type=str, default="crpo")
    p.add_argument("--fusion_modes", type=str, nargs="+", default=PIXEL_STATE_MODES,
                   choices=PIXEL_STATE_MODES)
    p.add_argument("--variants", type=str, nargs="+", default=["mlp", "gap"],
                   choices=FUSION_VARIANTS,
                   help="Variants to compare; add 'base' for the original runs without the MLP.")
    p.add_argument("--last_k", type=int, default=10,
                   help="Logged points averaged to get each run's final value.")
    p.add_argument("--x_max", type=float, default=None,
                   help="Curves only: upper x limit (env steps); defaults to the data range.")
    p.add_argument("--no_threshold", action="store_true", help="Hide safety threshold lines.")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())
