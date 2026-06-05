"""Plot iMCTS BlackBox results compared against SRBench baselines.

Usage:  python -m imcts.benchmarks.plot

Reads iMCTS CSV results from  benchmark_results/imcts/blackbox/,
combines them with the SRBench feather file from
benchmark_results/srbench/black-box_results.feather, and writes
comparison figures to  assets/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default paths (relative to the MCTS-4-SR repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMCTS_DIR = REPO_ROOT / "benchmark_results" / "imcts" / "blackbox"
DEFAULT_SR_RESULTS = REPO_ROOT / "benchmark_results" / "srbench" / "black-box_results.feather"
DEFAULT_FIG_DIR = REPO_ROOT / "assets"
DEFAULT_SUMMARY = DEFAULT_FIG_DIR / "imcts_blackbox_summary.csv"

# ---------------------------------------------------------------------------
# Symbolic regression algorithms (from SRBench)
# ---------------------------------------------------------------------------
SYMBOLIC_ALGS = {
    "AFP", "AFP_FE", "BSR", "DSR", "FFX", "FEAT", "EPLEX",
    "GP-GOMEA", "gplearn", "ITEA", "MRGP", "Operon", "SBP-GP",
    "AIFeynman", "iMCTS",
}


# ===================================================================
#  Pareto front helper (from  pareto_utils.py)
# ===================================================================
def _check_dominance(p1: tuple, p2: tuple) -> int:
    flag1 = 0
    flag2 = 0
    for o1, o2 in zip(p1, p2):
        if o1 < o2:
            flag1 = 1
        elif o1 > o2:
            flag2 = 1
    if flag1 == 1 and flag2 == 0:
        return 1
    elif flag1 == 0 and flag2 == 1:
        return -1
    return 0


def _pareto_front(obj1: np.ndarray, obj2: np.ndarray) -> list[int]:
    """Return indices of points on the Pareto front (minimise both)."""
    n = len(obj1)
    front = []
    for i in range(n):
        p = (obj1[i], obj2[i])
        dcount = 0
        for j in range(n):
            q = (obj1[j], obj2[j])
            compare = _check_dominance(p, q)
            if compare == -1:
                dcount += 1
        if dcount == 0:
            front.append(i)
    f_obj2 = [obj2[f] for f in front]
    s2 = np.argsort(np.array(f_obj2))
    return [front[s] for s in s2]


# ===================================================================
#  I/O helpers
# ===================================================================
def read_imcts_results(imcts_dir: Path, algorithm_name: str = "iMCTS") -> pd.DataFrame:
    files = sorted(imcts_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {imcts_dir}")

    raw = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    expression = raw["simplified_expression"].fillna(raw["expression"])

    if "test_r2" not in raw.columns:
        print(f"  [WARNING] 'test_r2' column not found in CSV files; "
              f"available columns: {list(raw.columns)}")
        r2_values = np.nan
    else:
        r2_values = pd.to_numeric(raw["test_r2"], errors="coerce")

    imcts = pd.DataFrame({
        "dataset": raw["case_name"],
        "algorithm": algorithm_name,
        "random_state": raw["seed"],
        "training time (s)": raw["time_sec"],
        "model_size": raw["complexity"],
        "symbolic_model": expression,
        "mse_test": np.nan,
        "mae_test": np.nan,
        "r2_test": r2_values,
        "params_str": "imcts",
    })
    # Report rows where test_r2 is missing
    missing_mask = imcts["r2_test"].isna()
    if missing_mask.any():
        missing_datasets = imcts.loc[missing_mask, "dataset"].tolist()
        missing_seeds = imcts.loc[missing_mask, "random_state"].tolist()
        print(f"  [WARNING] {missing_mask.sum()} row(s) have missing test_r2:")
        for ds, seed in zip(missing_datasets, missing_seeds):
            print(f"    - dataset={ds}, seed={seed}")

    imcts["training time (hr)"] = imcts["training time (s)"] / 3600
    imcts["r2_zero_test"] = imcts["r2_test"].clip(lower=0)
    imcts["friedman_dataset"] = imcts["dataset"].str.contains("_fri_", regex=False)
    imcts["symbolic_alg"] = True
    return imcts


def merge_results(sr_results: Path, imcts: pd.DataFrame,
                  algorithm_name: str = "iMCTS") -> pd.DataFrame:
    srbench = pd.read_feather(sr_results)
    srbench = srbench[srbench["algorithm"] != algorithm_name].copy()
    combined = pd.concat([srbench, imcts], ignore_index=True)
    return combined


# ===================================================================
#  Summarisation
# ===================================================================
def summarize_by_dataset(combined: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "r2_test", "r2_zero_test", "model_size",
        "training time (s)", "training time (hr)",
    ]
    clean = combined.dropna(subset=["dataset", "algorithm", "r2_test", "model_size"])
    by_dataset = (clean.groupby(["algorithm", "dataset"], as_index=False)[metric_cols]
                  .median().copy())
    by_dataset["r2_test_rank"] = by_dataset.groupby("dataset")["r2_test"].rank(
        ascending=False, method="average")
    by_dataset["model_size_rank"] = by_dataset.groupby("dataset")["model_size"].rank(
        ascending=True, method="average")
    by_dataset["training_time_rank"] = by_dataset.groupby("dataset")[
        "training time (s)"].rank(ascending=True, method="average")
    return by_dataset


def bootstrap_interval(values: np.ndarray, n: int = 1000,
                       seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def build_summary(by_dataset: pd.DataFrame) -> pd.DataFrame:
    return (
        by_dataset.groupby("algorithm", as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            median_r2=("r2_test", "median"),
            mean_r2_rank=("r2_test_rank", "mean"),
            median_model_size=("model_size", "median"),
            mean_model_size_rank=("model_size_rank", "mean"),
            median_training_time_s=("training time (s)", "median"),
            mean_training_time_rank=("training_time_rank", "mean"),
        )
        .sort_values(["mean_r2_rank", "mean_model_size_rank"])
    )


# ===================================================================
#  Label overlap avoidance
# ===================================================================
def _avoid_label_overlaps(xy_labels, spacing=1.2, max_iter=80,
                         labels=None, ha_list=None):
    """Simple force-directed adjustment to reduce label overlaps.

    Parameters
    ----------
    xy_labels : list of (x, y) tuples
        Initial label *anchor* positions in data coordinates.
    spacing : float
        Minimum desired distance between label visual centres.
    max_iter : int
        Maximum number of iterations.
    labels : list of str, optional
        Label text strings, used to estimate visual extent.
    ha_list : list of str, optional
        Horizontal alignment for each label (``'left'``, ``'right'``,
        or ``'center'``).  When provided together with *labels*, the
        algorithm converts anchors to estimated visual centres before
        de-overlapping and maps them back afterwards.

    Returns
    -------
    list of (x, y) tuples – adjusted *anchor* positions.
    """
    n = len(xy_labels)
    if n <= 1:
        return xy_labels

    anchors = np.array(xy_labels, dtype=float)

    # ---- convert anchors → visual centres when ha info is available ----
    if labels is not None and ha_list is not None and len(labels) == n:
        # rough char width in data coordinates (depends on fontsize, axes
        # limits, figure size, … – a heuristic, but sufficient for relative
        # de-overlap)
        char_w = 0.13
        centres = anchors.copy()
        for i, (ha, lbl) in enumerate(zip(ha_list, labels)):
            half_w = len(lbl) * char_w / 2
            if ha == "left":
                centres[i, 0] += half_w  # text extends right of anchor
            elif ha == "right":
                centres[i, 0] -= half_w  # text extends left of anchor
        pos = centres.copy()
    else:
        pos = anchors.copy()

    # ---- force-directed repulsion on (visual) centres ----
    for _ in range(max_iter):
        max_force = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = float(np.linalg.norm(diff))
                if dist < spacing and dist > 1e-10:
                    direction = diff / dist
                    push = (spacing - dist) * 0.35
                    pos[i] += direction * push
                    pos[j] -= direction * push
                    max_force = max(max_force, push)
        # spring back toward original centres
        if labels is not None and ha_list is not None:
            pos += (centres - pos) * 0.12
        else:
            pos += (anchors - pos) * 0.12
        if max_force < 1e-4:
            break

    # ---- convert centres back to anchors when ha info was used ----
    if labels is not None and ha_list is not None and len(labels) == n:
        char_w = 0.13
        for i, (ha, lbl) in enumerate(zip(ha_list, labels)):
            half_w = len(lbl) * char_w / 2
            if ha == "left":
                pos[i, 0] -= half_w
            elif ha == "right":
                pos[i, 0] += half_w

    return [tuple(p) for p in pos]


# ===================================================================
#  Drawing
# ===================================================================
def save_pairgrid(by_dataset: pd.DataFrame, fig_dir: Path) -> Path:
    """SRBench-style pairgrid: R² test, Model Size, Training Time."""
    x_vars = ["r2_test", "model_size", "training time (s)"]
    titles = ["$R^2$ Test", "Model Size", "Training Time (s)"]
    df_plot = by_dataset.dropna(subset=x_vars).copy()
    df_plot["algorithm_label"] = df_plot["algorithm"].apply(
        lambda name: f"*{name}" if name in SYMBOLIC_ALGS else name)

    order = (df_plot.groupby("algorithm_label")["r2_test"]
             .median().sort_values(ascending=False).index.to_list())
    y_pos = np.arange(len(order))
    colors = plt.cm.magma_r(np.linspace(0.15, 0.85, len(order)))

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 6.5), sharey=True)
    for ax, x_var, title in zip(axes, x_vars, titles):
        stats = []
        for i, algorithm_label in enumerate(order):
            values = df_plot.loc[df_plot["algorithm_label"] == algorithm_label, x_var].values
            median = float(np.nanmedian(values))
            ci_low, ci_high = bootstrap_interval(values, seed=42 + i)
            stats.append((median, ci_low, ci_high))

        medians = np.array([s[0] for s in stats])
        ci_lows = np.array([s[1] for s in stats])
        ci_highs = np.array([s[2] for s in stats])
        xerr = np.vstack([medians - ci_lows, ci_highs - medians])
        ax.errorbar(medians, y_pos, xerr=xerr, fmt="o", markersize=8,
                     linewidth=1, capsize=0, markeredgecolor="white",
                     markeredgewidth=0.8, color="#303030",
                     ecolor="#303030", zorder=2)
        ax.scatter(medians, y_pos, s=80, c=colors, edgecolors="white",
                    linewidths=0.8, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(axis="y", color="0.86", linewidth=0.9)
        ax.grid(axis="x", color="0.92", linewidth=0.8)
        if x_var == "r2_test":
            ax.set_xlim(-0.25, 1)
        else:
            ax.set_xscale("log")

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    path = fig_dir / "blackbox_pairgrid.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pareto_rank(by_dataset: pd.DataFrame, fig_dir: Path) -> Path:
    """Rank Pareto plot: R² test rank vs model size rank."""
    xcol, ycol = "r2_test_rank", "model_size_rank"
    data = by_dataset.dropna(subset=[xcol, ycol]).copy()
    data["algorithm_label"] = data["algorithm"].apply(
        lambda name: f"{name}*" if name in SYMBOLIC_ALGS else name)
    pareto_data = data.groupby("algorithm_label")[[xcol, ycol]].median()

    objs = pareto_data[[xcol, ycol]].values.copy()
    levels = 6
    styles = ["-", "-.", "--", ":", ":", ":"]
    pareto_ranks = -np.ones(len(pareto_data))
    pareto_fronts = []
    for level in range(levels):
        front = _pareto_front(objs[:, 0], objs[:, 1])
        if len(front) > 0:
            pareto_ranks[front] = level
        objs[front, :] = np.inf
        pareto_fronts.append(front)

    pareto_data = pareto_data.copy()
    pareto_data["pareto_rank"] = pareto_ranks
    rank_count = max(1, int(pareto_data["pareto_rank"].nunique()))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, rank_count))

    fig, ax = plt.subplots(figsize=(7, 7))
    for i, front in enumerate(pareto_fronts):
        if not len(front):
            continue
        front_data = pareto_data.iloc[front]
        ax.plot(front_data[xcol], front_data[ycol],
                styles[min(i, len(styles) - 1)],
                color="black", alpha=0.5, zorder=1)

    for _, row in pareto_data.iterrows():
        rank = int(row["pareto_rank"])
        rank = max(0, rank)
        color = cmap[min(rank, len(cmap) - 1)]
        ax.scatter(row[xcol], row[ycol], s=250, color=color, zorder=2)

    # -- compute initial label positions before overlap avoidance --
    label_specs = []  # (x, y, label, ha, va)
    for algorithm_label, row in pareto_data.iterrows():
        x = row[xcol] - 0.5
        y = row[ycol] - 0.3
        ha = "right"
        va = "top"
        if algorithm_label in ["Linear", "AFP_FE*", "MLP", "MRGP*", "iMCTS*"]:
            x = row[xcol] + 0.5
            ha = "left"
        elif algorithm_label == "Operon*":
            x = row[xcol]
            y = row[ycol] + 0.8
            ha = "center"
            va = "bottom"
        elif algorithm_label in ["gplearn*", "FEAT*"]:
            y = row[ycol] + 1
        label_specs.append((x, y, algorithm_label, ha, va))

    # de-overlap  (pass labels + ha so visual centres are used)
    anchors = [(s[0], s[1]) for s in label_specs]
    lbl_texts = [s[2] for s in label_specs]
    ha_vals = [s[3] for s in label_specs]
    adjusted = _avoid_label_overlaps(
        anchors, spacing=1.5, labels=lbl_texts, ha_list=ha_vals)

    for (x, y), (_, _, alg_label, ha, va) in zip(adjusted, label_specs):
        ax.text(x, y, alg_label, ha=ha, va=va,
                bbox={"facecolor": "white", "edgecolor": "blue",
                      "boxstyle": "round", "alpha": 1})

    for algorithm_label, group in data.groupby("algorithm_label"):
        x = group[xcol].median()
        y_val = group[ycol].median()
        ci_low_x, ci_high_x = bootstrap_interval(group[xcol].values)
        ci_low_y, ci_high_y = bootstrap_interval(group[ycol].values)
        rank = int(pareto_data.loc[algorithm_label, "pareto_rank"])
        color = cmap[min(max(rank, 0), len(cmap) - 1)]
        ax.plot([ci_low_x, ci_high_x], [y_val, y_val], alpha=0.5, color=color)
        ax.plot([x, x], [ci_low_y, ci_high_y], alpha=0.5, color=color)

    ax.set_aspect(1.0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=16)
    ax.set_xlabel("Median $R^2$ Test Rank", fontsize=18)
    ax.set_ylabel("Median Model Size Rank", fontsize=18)
    ax.grid(color="0.88")
    fig.tight_layout()
    path = fig_dir / "blackbox_pareto_rank.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def save_accuracy_complexity(summary: pd.DataFrame, fig_dir: Path,
                             algorithm_name: str = "iMCTS") -> Path:
    """Scatter: mean R² rank vs mean model-size rank, highlighting iMCTS."""
    summary = summary.copy()
    is_focus = summary["algorithm"] == algorithm_name

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(summary.loc[~is_focus, "mean_r2_rank"],
               summary.loc[~is_focus, "mean_model_size_rank"],
               s=62, color="#5b6770", alpha=0.82, label="SRBench")
    ax.scatter(summary.loc[is_focus, "mean_r2_rank"],
               summary.loc[is_focus, "mean_model_size_rank"],
               s=115, color="#d62728", marker="D",
               label=algorithm_name, zorder=3)
    ax.invert_xaxis()
    ax.invert_yaxis()
    # -- place labels with overlap avoidance --
    label_specs = []  # (x, y, label, color, weight)
    for row in summary.itertuples(index=False):
        color = "#d62728" if row.algorithm == algorithm_name else "0.25"
        weight = "bold" if row.algorithm == algorithm_name else "normal"
        label_specs.append(
            (row.mean_r2_rank + 0.08, row.mean_model_size_rank + 0.08,
             row.algorithm, color, weight))

    adjusted = _avoid_label_overlaps(
        [(s[0], s[1]) for s in label_specs], spacing=1.2)

    for (x, y), (_, _, alg, color, weight) in zip(adjusted, label_specs):
        ax.text(x, y, alg, fontsize=8, color=color, weight=weight)
    ax.set_xlabel("Mean dataset rank for test $R^2$ (lower is better)")
    ax.set_ylabel("Mean dataset rank for model size (lower is simpler)")
    ax.set_title("BlackBox accuracy-complexity trade-off")
    ax.grid(alpha=0.25)
    ax.legend(title="")
    fig.tight_layout()
    path = fig_dir / "blackbox_accuracy_complexity_rank.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def save_r2_distribution(by_dataset: pd.DataFrame, fig_dir: Path) -> Path:
    """Boxplot of R² per algorithm."""
    order = (by_dataset.groupby("algorithm")["r2_test"]
             .median().sort_values(ascending=False).index)
    height = max(7, 0.32 * len(order))
    data = [by_dataset.loc[by_dataset["algorithm"] == alg, "r2_test"].values
            for alg in order]

    fig, ax = plt.subplots(figsize=(10, height))
    ax.boxplot(data, vert=False, tick_labels=order, patch_artist=True,
               flierprops={"markersize": 2, "marker": "o", "alpha": 0.35},
               boxprops={"facecolor": "#7fb3d5", "edgecolor": "#2f4f5f"},
               medianprops={"color": "#1f2d35", "linewidth": 1.6},
               whiskerprops={"color": "#2f4f5f"},
               capprops={"color": "#2f4f5f"})
    plt.axvline(0, color="0.6", linewidth=1)
    ax.set_xlim(-0.5, 1.05)
    ax.set_xlabel("Test $R^2$")
    ax.set_ylabel("")
    ax.set_title("BlackBox test accuracy by algorithm")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = fig_dir / "blackbox_r2_distribution.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def save_r2_rank_plot(summary: pd.DataFrame, fig_dir: Path) -> Path:
    """Horizontal scatter of mean R² rank."""
    order = summary.sort_values("mean_r2_rank")["algorithm"]
    height = max(7, 0.32 * len(order))
    plot_data = summary.set_index("algorithm").loc[order].reset_index()
    y_pos = np.arange(len(plot_data))

    fig, ax = plt.subplots(figsize=(9, height))
    ax.scatter(plot_data["mean_r2_rank"], y_pos, color="#1f77b4", s=58)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["algorithm"])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Mean dataset rank for test $R^2$ (lower is better)")
    ax.set_ylabel("")
    ax.set_title("BlackBox accuracy ranking")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = fig_dir / "blackbox_r2_rank.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


# ===================================================================
#  Main
# ===================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare iMCTS BlackBox results against SRBench baselines.")
    parser.add_argument("--imcts-dir", type=Path, default=DEFAULT_IMCTS_DIR,
                        help="Directory with one iMCTS CSV per BlackBox problem.")
    parser.add_argument("--sr-results", type=Path, default=DEFAULT_SR_RESULTS,
                        help="SRBench BlackBox feather file.")
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR,
                        help="Directory where figures are written.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY,
                        help="Where to write the algorithm-level summary CSV.")
    parser.add_argument("--algorithm-name", default="iMCTS",
                        help="Display name for the iMCTS algorithm.")
    parser.add_argument("--no-extra", action="store_true",
                        help="Skip auxiliary figures (distribution, rank, acc-comp).")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    matplotlib.rc("pdf", fonttype=42)
    plt.rcParams.update({
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

    # --- Load & merge -------------------------------------------------------
    imcts = read_imcts_results(args.imcts_dir, args.algorithm_name)
    combined = merge_results(args.sr_results, imcts, args.algorithm_name)

    # --- Summarise -----------------------------------------------------------
    by_dataset = summarize_by_dataset(combined)
    summary = build_summary(by_dataset)
    summary.to_csv(args.summary_csv, index=False)

    # --- Main figures --------------------------------------------------------
    p1 = save_pairgrid(by_dataset, args.fig_dir)
    print(f"Pairgrid:  {p1}")
    p2 = save_pareto_rank(by_dataset, args.fig_dir)
    print(f"Pareto:    {p2}")

    # --- Extra figures -------------------------------------------------------
    if not args.no_extra:
        p3 = save_r2_distribution(by_dataset, args.fig_dir)
        print(f"R² dist:   {p3}")
        p4 = save_r2_rank_plot(summary, args.fig_dir)
        print(f"R² rank:   {p4}")
        p5 = save_accuracy_complexity(summary, args.fig_dir, args.algorithm_name)
        print(f"Acc-comp:  {p5}")

    # --- Report --------------------------------------------------------------
    imcts_missing = int(imcts["r2_test"].isna().sum())
    print(f"\nRead {len(imcts)} {args.algorithm_name} rows from {args.imcts_dir}")
    print(f"{args.algorithm_name} datasets: {imcts['dataset'].nunique()}")
    print(f"Rows with missing r2_test: {imcts_missing}")
    print(f"Combined rows: {len(combined)}")
    print(f"Combined datasets: {combined['dataset'].nunique()}")
    print(f"Summary saved to {args.summary_csv}")
    print(f"All figures saved to {args.fig_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
