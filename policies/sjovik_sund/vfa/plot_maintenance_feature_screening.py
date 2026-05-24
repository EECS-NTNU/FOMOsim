#!/usr/bin/env python3
"""
plot_maintenance_feature_screening.py

Create thesis figures from maintenance_feature_screening.py outputs:
  - diagnostic traffic-light plot
  - feature-target correlation heatmap
  - maintenance-feature redundancy heatmap
  - VIF score bar plot
  - sparsity bar plot

Example:
    python policies/sjovik_sund/vfa/plot_maintenance_feature_screening.py \
      --input_dir models/maintenance_feature_screening_greedy_5ep_14d
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "fomosim_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/private/tmp") / "fomosim_cache"))

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

from policies.sjovik_sund.vfa.vfa_features import get_feature_short_name

# Thesis plot style shared with hyperparameter_tuning.ipynb.
BASE_FONTSIZE = 12
TITLE_FONTSIZE = 16
AXIS_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
ANNOT_FONTSIZE = 11
SMALL_FONTSIZE = 10
HEATMAP_FONTSIZE = 11
PLOT_DPI = 300

THESIS_USETEX = shutil.which("latex") is not None
THESIS_SERIF_FONTS = ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"]
THESIS_SANS_FONTS = ["Latin Modern Sans", "DejaVu Sans"]
THESIS_MONO_FONTS = ["Latin Modern Mono", "Latin Modern Typewriter", "DejaVu Sans Mono"]

plt.rcParams.update({
    "text.usetex": THESIS_USETEX,
    "font.family": "serif",
    "font.serif": THESIS_SERIF_FONTS,
    "font.sans-serif": THESIS_SANS_FONTS,
    "font.monospace": THESIS_MONO_FONTS,
    "mathtext.fontset": "cm",
})
if THESIS_USETEX:
    plt.rcParams["text.latex.preamble"] = r"\usepackage[T1]{fontenc} \usepackage{mlmodern}"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": THESIS_SERIF_FONTS,
    "font.sans-serif": THESIS_SANS_FONTS,
    "font.monospace": THESIS_MONO_FONTS,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "font.size": BASE_FONTSIZE,
    "font.weight": "normal",
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": AXIS_FONTSIZE,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "legend.title_fontsize": LEGEND_FONTSIZE,
    "figure.titlesize": TITLE_FONTSIZE,
    "figure.titleweight": "normal",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
})

# Shared divergent palette: blue (low) -> white (zero) -> red (high).
DIVERGENT_CMAP = LinearSegmentedColormap.from_list(
    "custom_palette", ["#3758d8", "#ffffff", "#db3249"]
)

# Categorical traffic-light colors derived from the same palette.
PASS_COLOR = "#3758d8"
FLAG_COLOR = "#db3249"
WARN_COLOR = "#d89c2b"


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


def _feature_labels(labels):
    return [get_feature_short_name(str(label)) for label in labels]


def _load_summary(input_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(input_dir / "feature_screening_summary.csv")
    first = summary.columns[0]
    if first != "feature":
        summary = summary.rename(columns={first: "feature"})
    return summary


def _maintenance_features(summary: pd.DataFrame) -> list[str]:
    if "screening_group" in summary.columns:
        features = summary.loc[summary["screening_group"].eq("maintenance"), "feature"].tolist()
        if features:
            return features
    features = summary[summary["family"].notna()]["feature"].tolist()
    if not features:
        # Fallback if metadata is missing.
        base = {
            "rebalancing_imbalance",
            "squared_starvation_penalty",
            "squared_congestion_penalty",
            "starvation_count",
            "congestion_count",
            "gross_starvation_risk",
            "gross_congestion_risk",
        }
        features = [f for f in summary["feature"].tolist() if f not in base]
    return features


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
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
        if col not in df:
            matrix.append(np.ones(len(df)))
            continue
        passed = df[col].fillna(False).astype(bool).eq(expected_pass_value)
        matrix.append(passed.astype(float).to_numpy())
    data = np.vstack(matrix).T

    fig_h = max(5.0, 0.34 * len(features) + 1.6)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    ax.imshow(data, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([FLAG_COLOR, PASS_COLOR]), vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(_feature_labels(features), fontsize=TICK_FONTSIZE)
    ax.set_title("Maintenance feature screening diagnostics", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Diagnostic")
    ax.set_ylabel("Candidate maintenance feature")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "pass" if data[i, j] else "flag", ha="center", va="center",
                    fontsize=SMALL_FONTSIZE, color="white")

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
    ax.set_yticklabels(_feature_labels(features), fontsize=TICK_FONTSIZE)
    ax.set_title("Feature-target correlations", fontsize=TITLE_FONTSIZE)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            text = "" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=SMALL_FONTSIZE, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("Pearson correlation")
    _save(fig, output_dir, "02_feature_target_correlation_heatmap")


def plot_feature_correlation_heatmap(input_dir: Path, features: list[str], output_dir: Path, method: str) -> None:
    fname = "feature_spearman_correlations.csv" if method == "spearman" else "feature_pearson_correlations.csv"
    corr = pd.read_csv(input_dir / fname, index_col=0)
    data = corr.reindex(index=features, columns=features).astype(float)

    display_df = data.copy()
    display_df.index = _feature_labels(display_df.index)
    display_df.columns = _feature_labels(display_df.columns)

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
        annot_kws={"fontsize": SMALL_FONTSIZE},
    )
    plt.title(f"Maintenance feature correlation matrix ({method})", fontsize=TITLE_FONTSIZE)
    plt.xticks(rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    plt.yticks(rotation=0, fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    _save(fig, output_dir, f"03_maintenance_feature_{method}_heatmap")


def plot_vif(summary: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    df = summary.set_index("feature").reindex(features)
    df = df[["vif"]].dropna().sort_values("vif", ascending=True)

    fig_h = max(4.5, 0.32 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    colors = np.where(df["vif"] > 20.0, FLAG_COLOR, np.where(df["vif"] > 10.0, WARN_COLOR, PASS_COLOR))
    ax.barh(_feature_labels(df.index), df["vif"], color=colors)
    ax.axvline(10.0, color=WARN_COLOR, linestyle="--", linewidth=1.2, label="vif = 10")
    ax.axvline(20.0, color=FLAG_COLOR, linestyle="--", linewidth=1.2, label="vif = 20")
    ax.set_xlabel("Variance inflation factor")
    ax.set_title("Maintenance feature multicollinearity")
    ax.legend(loc="lower right")
    _save(fig, output_dir, "04_vif_scores")


def plot_sparsity(summary: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    df = summary.set_index("feature").reindex(features)
    df = df[["fraction_zero"]].dropna().sort_values("fraction_zero", ascending=True)

    fig_h = max(4.5, 0.32 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    colors = np.where(df["fraction_zero"] > 0.95, FLAG_COLOR, np.where(df["fraction_zero"] > 0.80, WARN_COLOR, PASS_COLOR))
    ax.barh(_feature_labels(df.index), df["fraction_zero"], color=colors)
    ax.axvline(0.95, color=FLAG_COLOR, linestyle="--", linewidth=1.2, label="sparsity threshold = 0.95")
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of candidate states where feature is zero")
    ax.set_title("maintenance feature sparsity")
    ax.legend(loc="lower right")
    _save(fig, output_dir, "05_fraction_zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot maintenance feature-screening diagnostics.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="models/maintenance_feature_screening_greedy_10ep_21d",
        help="Directory produced by maintenance_feature_screening.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Figure directory. Defaults to <input_dir>/plots.",
    )
    parser.add_argument(
        "--include_base",
        action="store_true",
        help="Include base rebalancing features in the plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"

    summary = _load_summary(input_dir)
    features = summary["feature"].tolist() if args.include_base else _maintenance_features(summary)

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
