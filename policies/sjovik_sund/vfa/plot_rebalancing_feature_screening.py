#!/usr/bin/env python3
"""
plot_rebalancing_feature_screening.py

Create thesis figures from rebalancing_feature_screening.py outputs.
The generated figures match the maintenance screening visual style.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "fomosim_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/private/tmp") / "fomosim_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# Match the LaTeX styling used in plot_feature_study.py.
if shutil.which("latex"):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{mlmodern}",
    })
else:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
    })

# Shared divergent palette: blue (low) -> white (zero) -> red (high).
DIVERGENT_CMAP = LinearSegmentedColormap.from_list(
    "custom_palette", ["#3758d8", "#ffffff", "#db3249"]
)

# Categorical traffic-light colors derived from the same palette.
PASS_COLOR = "#3758d8"
FLAG_COLOR = "#db3249"
WARN_COLOR = "#d89c2b"


def _strip_underscores(labels):
    return [str(label).replace("_", " ") for label in labels]


DIAGNOSTIC_COLUMNS = [
    ("std ok", "near_constant", False),
    ("not sparse", "sparse", False),
    ("scale ok", "poor_scaling", False),
    ("target signal", "weak_signal", False),
    ("vif ok", "vif_flag", False),
    ("vif safe", "vif_serious", False),
]

TARGET_COLUMNS = [
    "corr_reward",
    "corr_td_target",
    "corr_future_starvation",
    "corr_future_congestion",
    "corr_future_broken_fleet_ratio",
]


def _load_summary(input_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(input_dir / "feature_screening_summary.csv")
    first = summary.columns[0]
    if first != "feature":
        summary = summary.rename(columns={first: "feature"})
    return summary


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=220, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


def plot_diagnostic_traffic_light(summary: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    df = summary.set_index("feature").loc[features]
    matrix = []
    labels = []
    for label, col, expected_pass_value in DIAGNOSTIC_COLUMNS:
        labels.append(label)
        passed = df[col].fillna(False).astype(bool).eq(expected_pass_value)
        matrix.append(passed.astype(float).to_numpy())
    data = np.vstack(matrix).T

    fig_h = max(5.0, 0.34 * len(features) + 1.6)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    ax.imshow(data, aspect="auto", interpolation="nearest", cmap=ListedColormap([FLAG_COLOR, PASS_COLOR]), vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(_strip_underscores(features), fontsize=8)
    ax.set_title("Rebalancing feature screening diagnostics")
    ax.set_xlabel("Diagnostic")
    ax.set_ylabel("Candidate rebalancing feature")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "pass" if data[i, j] else "flag", ha="center", va="center", fontsize=6, color="white")
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(features), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    _save(fig, output_dir, "01_diagnostic_traffic_light")


def plot_target_correlation_heatmap(input_dir: Path, features: list[str], output_dir: Path) -> None:
    corr = pd.read_csv(input_dir / "feature_target_correlations.csv").set_index("feature")
    cols = [c for c in TARGET_COLUMNS if c in corr.columns]
    data = corr.reindex(features)[cols].astype(float)

    fig_h = max(5.0, 0.34 * len(features) + 1.7)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    im = ax.imshow(data.to_numpy(), aspect="auto", interpolation="nearest", cmap=DIVERGENT_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("corr_", "").replace("_", " ") for c in cols], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(_strip_underscores(features), fontsize=8)
    ax.set_title("Rebalancing feature-target correlations")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            ax.text(j, i, "" if pd.isna(val) else f"{val:.2f}", ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("Pearson correlation")
    _save(fig, output_dir, "02_feature_target_correlation_heatmap")


def plot_feature_correlation_heatmap(input_dir: Path, features: list[str], output_dir: Path, method: str = "spearman") -> None:
    fname = "feature_spearman_correlations.csv" if method == "spearman" else "feature_pearson_correlations.csv"
    corr = pd.read_csv(input_dir / fname, index_col=0)
    data = corr.reindex(index=features, columns=features).astype(float)

    display_df = data.copy()
    display_df.index = display_df.index.astype(str).str.replace("_", " ")
    display_df.columns = display_df.columns.astype(str).str.replace("_", " ")

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        display_df,
        annot=True,
        cmap=DIVERGENT_CMAP,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1.0,
        vmax=1.0,
        annot_kws={"fontweight": "bold"},
    )
    plt.title(f"Rebalancing feature redundancy ({method})", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save(fig, output_dir, f"03_rebalancing_feature_{method}_heatmap")


def plot_vif(summary: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    df = summary.set_index("feature").reindex(features)[["vif"]].dropna().sort_values("vif", ascending=True)
    fig_h = max(4.5, 0.32 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    colors = np.where(df["vif"] > 10.0, FLAG_COLOR, np.where(df["vif"] > 5.0, WARN_COLOR, PASS_COLOR))
    ax.barh(_strip_underscores(df.index), df["vif"], color=colors)
    ax.axvline(5.0, color=WARN_COLOR, linestyle="--", linewidth=1.2, label="vif = 5")
    ax.axvline(10.0, color=FLAG_COLOR, linestyle="--", linewidth=1.2, label="vif = 10")
    ax.set_xlabel("Variance inflation factor")
    ax.set_title("Rebalancing feature multicollinearity")
    ax.legend(loc="lower right")
    _save(fig, output_dir, "04_vif_scores")


def plot_sparsity(summary: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    df = summary.set_index("feature").reindex(features)[["fraction_zero"]].dropna().sort_values("fraction_zero", ascending=True)
    fig_h = max(4.5, 0.32 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    colors = np.where(df["fraction_zero"] > 0.95, FLAG_COLOR, np.where(df["fraction_zero"] > 0.80, WARN_COLOR, PASS_COLOR))
    ax.barh(_strip_underscores(df.index), df["fraction_zero"], color=colors)
    ax.axvline(0.95, color=FLAG_COLOR, linestyle="--", linewidth=1.2, label="sparsity threshold = 0.95")
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of candidate states where feature is zero")
    ax.set_title("rebalancing feature sparsity")
    ax.legend(loc="lower right")
    _save(fig, output_dir, "05_fraction_zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot rebalancing feature-screening diagnostics.")
    parser.add_argument("--input_dir", type=str, default="models/feature_study_v2")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    summary = _load_summary(input_dir)
    features = summary["feature"].tolist()

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Features plotted: {len(features)}")
    plot_diagnostic_traffic_light(summary, features, output_dir)
    plot_target_correlation_heatmap(input_dir, features, output_dir)
    plot_feature_correlation_heatmap(input_dir, features, output_dir, method="pearson")
    plot_feature_correlation_heatmap(input_dir, features, output_dir, method="spearman")
    plot_vif(summary, features, output_dir)
    plot_sparsity(summary, features, output_dir)


if __name__ == "__main__":
    main()
