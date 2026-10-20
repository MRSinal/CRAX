"""Compare early-fusion techniques for combining pixels and proprioception.

Two kinds of figure are written:

  * Learning curves (`<out_name>_curves.pdf`): one row per environment, and in each
    row a separate Reward | Cost pair per difficulty level, one line per fusion mode
    (mean over seeds with a 95% CI band).
  * Final-performance bars (`<out_name>_<env>.pdf`): one figure per environment. Each figure is a grid with the metrics down the rows
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
    moving_average,
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


def plot_curves(store: RunStore, args: argparse.Namespace) -> Optional[Path]:
    """Learning curves for every env, with Level 1 and Level 2 as separate panel pairs.

    Rows are environments; each row holds one subfigure per level, titled
    "<env> - Level <l>", containing the metrics side by side (Reward | Cost).
    """
    set_mpl_style()
    envs, metrics, levels, modes = args.envs, args.metrics, args.levels, args.fusion_modes

    n_rows, n_cols = len(envs), len(levels) * len(metrics)
    fig = plt.figure(figsize=(args.panel_w * n_cols, args.panel_h * n_rows + 0.6),
                     layout="constrained")
    subfigs = np.atleast_2d(fig.subfigures(n_rows, len(levels), wspace=0.04, hspace=0.04))

    handles: Dict[str, plt.Artist] = {}
    drew_any = False

    for env_i, env in enumerate(envs):
        for level_i, level in enumerate(levels):
            sub = subfigs[env_i][level_i]
            sub.suptitle(f"{TRANSLATIONS.get(env, env)} - Level {level}")
            axs = np.atleast_1d(sub.subplots(1, len(metrics)))

            for metric_i, metric in enumerate(metrics):
                ax = axs[metric_i]
                non_negative = True

                for mode in modes:
                    runs = store.get((env, level, mode, metric), [])
                    if not runs:
                        continue
                    steps, vals = align_and_stack(runs)
                    if vals.size == 0:
                        continue
                    drew_any = True
                    mode_non_negative = float(vals.min()) >= 0.0
                    non_negative &= mode_non_negative

                    x = steps.astype(float)
                    mean = vals.mean(axis=0)
                    ci = 1.96 * vals.std(axis=0) / np.sqrt(max(vals.shape[0], 1))
                    if args.smoothing_window:
                        mean = moving_average(mean, args.smoothing_window)
                        ci = moving_average(ci, args.smoothing_window)

                    lower = mean - ci
                    if mode_non_negative:
                        lower = np.maximum(lower, 0.0)

                    line, = ax.plot(x, mean, color=FUSION_MODE_COLORS.get(mode))
                    ax.fill_between(x, lower, mean + ci, alpha=0.25, color=line.get_color())
                    handles.setdefault(mode, line)

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
        print("No curve data, skipping learning-curve figure.")
        return None

    labels, hs = zip(*handles.items())
    labels = [TRANSLATIONS.get(lbl, lbl) for lbl in labels]
    fig.legend(hs, labels, loc="outside lower center",
               ncol=legend_ncol(len(modes), len(labels)), fancybox=True, shadow=True)

    out_dir = results_path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_curves.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return out_path


def build_stats(store: RunStore, args: argparse.Namespace) -> pd.DataFrame:
    """Final mean/CI per (env, level, mode, metric), keyed as 'algo' for reuse.

    The win/safety helpers from `alg_comp_bars` read an 'algo' column; we put the
    fusion mode there so a mode plays the role an algorithm does in that table.
    """
    rows = []
    for env in args.envs:
        for level in args.levels:
            for mode in args.fusion_modes:
                for metric in args.metrics:
                    stat = _final_value(store.get((env, level, mode, metric), []), args.last_k)
                    if stat is None:
                        continue
                    mean, ci = stat
                    rows.append(dict(env=env, level=level, algo=mode, metric=metric,
                                     mean=mean, ci=ci))
    return pd.DataFrame(rows)


Lookup = Dict[Tuple[str, int, str, str], Tuple[float, float]]


def _build_lookup(stats: pd.DataFrame) -> Lookup:
    """(env, level, mode, metric) -> (mean, ci)."""
    return {(r["env"], r["level"], r["algo"], r["metric"]): (r["mean"], r["ci"])
            for _, r in stats.iterrows()}


def _is_safe(lookup: Lookup, env: str, level: int, mode: str, threshold: float) -> bool:
    """Safe if the mean cost is below the threshold."""
    key = (env, level, mode, "cost")
    return key in lookup and lookup[key][0] < threshold


def _best_mode(lookup: Lookup, envs: List[str], levels: List[int], modes: List[str],
               threshold: Optional[float] = None) -> Dict[Tuple[str, int], Optional[str]]:
    """Highest-reward mode per (env, level); restricted to safe modes if `threshold` is set."""
    best: Dict[Tuple[str, int], Optional[str]] = {}
    for env in envs:
        for level in levels:
            cands = [m for m in modes if (env, level, m, "reward") in lookup
                     and (threshold is None or _is_safe(lookup, env, level, m, threshold))]
            best[(env, level)] = max(cands, key=lambda m: lookup[(env, level, m, "reward")][0],
                                     default=None)
    return best


def generate_fusion_summary_table(stats: pd.DataFrame, envs: List[str], levels: List[int],
                                  modes: List[str], safety_threshold: float) -> str:
    """Wins / Safe% per fusion mode, levels across the columns plus a Total.

    Mirrors `alg_comp_bars.generate_latex_summary_table` but rows are fusion modes:
      - Wins: envs where the mode had the highest reward while safe
      - Safe%: percentage of envs where the mode was safe
    """
    lookup = _build_lookup(stats)
    best_mode = _best_mode(lookup, envs, levels, modes, safety_threshold)

    summary = {}
    for mode in modes:
        for level in levels:
            wins = safe_count = total = 0
            for env in envs:
                if (env, level, mode, "cost") not in lookup:
                    continue
                total += 1
                if _is_safe(lookup, env, level, mode, safety_threshold):
                    safe_count += 1
                if best_mode.get((env, level)) == mode:
                    wins += 1
            safe_pct = (safe_count / total * 100) if total else 0
            summary[(mode, level)] = (wins, safe_pct)

    totals = {}
    for mode in modes:
        tw = 0
        pcts = []
        for level in levels:
            w, p = summary.get((mode, level), (0, 0))
            tw += w
            pcts.append(p)
        totals[mode] = (tw, sum(pcts) / len(pcts) if pcts else 0)

    max_wins = {level: max((summary.get((m, level), (0, 0))[0] for m in modes), default=0)
                for level in levels}
    max_total = max((totals[m][0] for m in modes), default=0)

    n_cols = (len(levels) + 1) * 2
    lines = [
        "% Auto-generated LaTeX fusion summary table",
        "% Requires: \\usepackage{booktabs}, \\usepackage{xcolor}",
        "\\definecolor{safegreen}{RGB}{34, 139, 34}",
        "",
        "\\begin{table*}[htbp]",
        "\\centering",
        f"\\begin{{tabular}}{{l{'c' * n_cols}}}",
        "\\toprule",
    ]

    header1 = [""] + [f"\\multicolumn{{2}}{{c}}{{Level {lvl}}}" for lvl in levels]
    header1.append("\\multicolumn{2}{c}{Total}")
    lines.append(" & ".join(header1) + " \\\\")

    cmid, col = [], 2
    for _ in levels:
        cmid.append(f"\\cmidrule(lr){{{col}-{col + 1}}}")
        col += 2
    cmid.append(f"\\cmidrule(lr){{{col}-{col + 1}}}")
    lines.append(" ".join(cmid))

    header2 = ["Fusion"]
    for _ in range(len(levels) + 1):  # per level + Total
        header2 += ["Wins $\\uparrow$", "Safe\\% $\\uparrow$"]
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")

    for mode in modes:
        parts = [TRANSLATIONS.get(mode, mode)]
        for level in levels:
            wins, safe_pct = summary.get((mode, level), (0, 0))
            w = f"\\textbf{{{wins}}}" if wins > 0 and wins == max_wins[level] else str(wins)
            s = f"{safe_pct:.0f}"
            if safe_pct == 100:
                s = f"\\textcolor{{safegreen}}{{{s}}}"
            parts += [w, s]
        tw, avg = totals[mode]
        tws = f"\\textbf{{{tw}}}" if tw > 0 and tw == max_total else str(tw)
        avgs = f"{avg:.0f}"
        if avg == 100:
            avgs = f"\\textcolor{{safegreen}}{{{avgs}}}"
        parts += [tws, avgs]
        lines.append(" & ".join(parts) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Early-fusion technique summary across {len(envs)} environments. "
        f"\\textbf{{Wins}}: number of environments where the fusion mode achieved the "
        f"highest reward while being safe. "
        f"\\textbf{{Safe\\%}}: percentage of environments where the mode was safe "
        f"(cost $< {safety_threshold:.0f}$). "
        f"\\textbf{{Total}}: sum of wins and average safety percentage across levels. "
        f"\\textcolor{{safegreen}}{{Green}} indicates 100\\% safe.}}",
        "\\label{tab:fusion_summary}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def generate_fusion_appendix_table(stats: pd.DataFrame, envs: List[str], levels: List[int],
                                   modes: List[str], safety_threshold: float,
                                   envs_per_row: int = 2) -> str:
    """Detailed per-mode reward/cost table, envs chunked across stacked sub-tables.

    Mirrors `alg_comp_bars.generate_latex_table_appendix` but rows are fusion modes:
    reward (R, higher better) and cost (C, lower better) per (env, level); safe costs
    green, the highest reward per (env, level) bold
    regardless of safety.
    """
    lookup = _build_lookup(stats)
    best_mode = _best_mode(lookup, envs, levels, modes)
    env_chunks = [envs[i:i + envs_per_row] for i in range(0, len(envs), envs_per_row)]

    mode_col_width, data_col_width = "1.8cm", "0.85cm"
    n_full_data_cols = envs_per_row * len(levels) * 2
    col_spec = f"p{{{mode_col_width}}}" + f"p{{{data_col_width}}}" * n_full_data_cols

    lines = [
        "% Auto-generated LaTeX fusion table (Appendix - split across rows)",
        "% Requires: \\usepackage{booktabs}, \\usepackage{xcolor}, \\usepackage{array}",
        "\\definecolor{safegreen}{RGB}{34, 139, 34}",
        "",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\footnotesize",
    ]

    for chunk_idx, env_chunk in enumerate(env_chunks):
        remaining = envs_per_row - len(env_chunk)
        if remaining:
            lines.append("\\noindent")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Row 1: env names
        row = [""]
        for env in env_chunk:
            row.append(f"\\multicolumn{{{len(levels) * 2}}}{{c}}{{{TRANSLATIONS.get(env, env)}}}")
        for _ in range(remaining):
            row.append(f"\\multicolumn{{{len(levels) * 2}}}{{c}}{{}}")
        lines.append(" & ".join(row) + " \\\\")

        cmid, col = [], 2
        for _ in env_chunk:
            cmid.append(f"\\cmidrule(lr){{{col}-{col + len(levels) * 2 - 1}}}")
            col += len(levels) * 2
        lines.append(" ".join(cmid))

        # Row 2: levels
        row = [""]
        for _ in env_chunk:
            for level in levels:
                row.append(f"\\multicolumn{{2}}{{c}}{{Level {level}}}")
        for _ in range(remaining):
            for _ in levels:
                row.append("\\multicolumn{2}{c}{}")
        lines.append(" & ".join(row) + " \\\\")

        cmid, col = [], 2
        for _ in env_chunk:
            for _ in levels:
                cmid.append(f"\\cmidrule(lr){{{col}-{col + 1}}}")
                col += 2
        lines.append(" ".join(cmid))

        # Row 3: R / C
        row = ["Fusion"]
        for _ in env_chunk:
            for _ in levels:
                row += ["R $\\uparrow$", "C $\\downarrow$"]
        for _ in range(remaining):
            for _ in levels:
                row += ["", ""]
        lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")

        # Data rows
        for mode in modes:
            row = [TRANSLATIONS.get(mode, mode)]
            for env in env_chunk:
                for level in levels:
                    rk, ck = (env, level, mode, "reward"), (env, level, mode, "cost")
                    if rk in lookup:
                        r = f"{lookup[rk][0]:.1f}"
                        if best_mode.get((env, level)) == mode:
                            r = f"\\textbf{{{r}}}"
                    else:
                        r = "--"
                    if ck in lookup:
                        c = f"{lookup[ck][0]:.1f}"
                        if _is_safe(lookup, env, level, mode, safety_threshold):
                            c = f"\\textcolor{{safegreen}}{{{c}}}"
                    else:
                        c = "--"
                    row += [r, c]
            for _ in range(remaining):
                for _ in levels:
                    row += ["", ""]
            lines.append(" & ".join(row) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        if chunk_idx < len(env_chunks) - 1:
            lines.append("\\\\[1em]")

    lines += [
        "\\caption{Detailed early-fusion comparison across environments and difficulty levels. "
        "R: Reward ($\\uparrow$ higher is better), C: Cost ($\\downarrow$ lower is better). "
        f"\\textcolor{{safegreen}}{{Green}} indicates safe (cost $< {safety_threshold:.0f}$). "
        "\\textbf{Bold} indicates the highest reward.}",
        "\\label{tab:fusion_comparison_detailed}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    store = load_all(args)
    if not any(store.values()):
        raise SystemExit(
            "No data loaded. Check --input/--envs/--algo/--levels/--seeds, and that the "
            "fusion runs were downloaded (python -m results.download.fusion_comparison)."
        )
    if not args.no_curves:
        plot_curves(store, args)
    if not args.no_bars:
        for env in args.envs:
            plot_env(store, args, env)

    if args.latex:
        stats = build_stats(store, args)
        summary = generate_fusion_summary_table(
            stats, args.envs, args.levels, args.fusion_modes, args.threshold)
        appendix = generate_fusion_appendix_table(
            stats, args.envs, args.levels, args.fusion_modes, args.threshold)
        print("\n% ===== SUMMARY TABLE =====\n" + summary)
        print("\n% ===== APPENDIX TABLE =====\n" + appendix + "\n")
        if args.output_latex:
            # Mirror alg_comp_bars: a base path yields <stem>_summary / <stem>_appendix.
            base = results_path(args.output_latex)
            base.parent.mkdir(parents=True, exist_ok=True)
            summary_path = base.with_stem(base.stem + "_summary")
            appendix_path = base.with_stem(base.stem + "_appendix")
            summary_path.write_text(summary)
            appendix_path.write_text(appendix)
            print(f"Summary table saved to: {summary_path}")
            print(f"Appendix table saved to: {appendix_path}")


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
    p.add_argument("--x_max", type=float, default=None,
                   help="Curves only: upper x limit (env steps); defaults to the data range.")
    p.add_argument("--no_curves", action="store_true", help="Skip the learning-curve figure.")
    p.add_argument("--no_bars", action="store_true",
                   help="Skip the per-env final-performance bar figures.")
    p.add_argument("--latex", action="store_true",
                   help="Also print a Wins/Safe%% summary LaTeX table (rows = fusion modes).")
    p.add_argument("--output_latex", type=str, default=None,
                   help="Path (relative to results/) to also write the LaTeX table to.")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())
