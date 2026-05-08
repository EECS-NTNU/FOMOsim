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
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "fomosim_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/private/tmp") / "fomosim_cache"))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


DIAGNOSTIC_COLUMNS = [
    ("std ok", "near_constant", False),
    ("not sparse", "sparse", False),
    ("scale ok", "poor_scaling", False),
    ("target signal", "weak_signal", False),
    ("VIF ok", "vif_flag", False),
    ("VIF safe", "vif_serious", False),
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


def _maintenance_features(summary: pd.DataFrame) -> list[str]:
    features = summary[summary["family"].notna()]["feature"].tolist()
    if not features:
        # Fallback if metadata is missing.
        base = {
            "rebalancing_imbalance",
            "squared_starvation_penalty",
            "squared_congestion_penalty",
            "gross_starvation_risk",
            "gross_congestion_risk",
        }
        features = [f for f in summary["feature"].tolist() if f not in base]
    return features


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")
    print(f"  wrote {svg}")


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
    cmap = ListedColormap(["#c93f3f", "#2f8f5b"])
    ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, fontsize=8)
    ax.set_title("Maintenance Feature Screening Diagnostics")
    ax.set_xlabel("Diagnostic")
    ax.set_ylabel("Candidate maintenance feature")

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
    im = ax.imshow(data.to_numpy(), aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("corr_", "").replace("_", " ") for c in cols], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, fontsize=8)
    ax.set_title("Feature-Target Correlations")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            text = "" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("Pearson correlation")
    _save(fig, output_dir, "02_feature_target_correlation_heatmap")


def plot_feature_correlation_heatmap(input_dir: Path, features: list[str], output_dir: Path, method: str) -> None:
    fname = "feature_spearman_correlations.csv" if method == "spearman" else "feature_pearson_correlations.csv"
    corr = pd.read_csv(input_dir / fname, index_col=0)
    data = corr.reindex(index=features, columns=features).astype(float)

    fig_w = max(8.0, 0.33 * len(features) + 2.4)
    fig_h = max(7.0, 0.33 * len(features) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data.to_numpy(), aspect="equal", interpolation="nearest", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=65, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, fontsize=7)
    ax.set_title(f"Maintenance Feature Redundancy ({method.title()})")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            if pd.isna(val):
                continue
            color = "white" if abs(float(val)) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label(f"{method.title()} correlation")
    _save(fig, output_dir, f"03_maintenance_feature_{method}_heatmap")


def plot_vif(summary: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    df = summary.set_index("feature").reindex(features)
    df = df[["vif"]].dropna().sort_values("vif", ascending=True)

    fig_h = max(4.5, 0.32 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    colors = np.where(df["vif"] > 20.0, "#c93f3f", np.where(df["vif"] > 10.0, "#d89c2b", "#2f8f5b"))
    ax.barh(df.index, df["vif"], color=colors)
    ax.axvline(10.0, color="#d89c2b", linestyle="--", linewidth=1.2, label="VIF = 10")
    ax.axvline(20.0, color="#c93f3f", linestyle="--", linewidth=1.2, label="VIF = 20")
    ax.set_xlabel("Variance inflation factor")
    ax.set_title("Maintenance Feature Multicollinearity")
    ax.legend(loc="lower right")
    _save(fig, output_dir, "04_vif_scores")


def plot_sparsity(summary: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    df = summary.set_index("feature").reindex(features)
    df = df[["fraction_zero"]].dropna().sort_values("fraction_zero", ascending=True)

    fig_h = max(4.5, 0.32 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    colors = np.where(df["fraction_zero"] > 0.95, "#c93f3f", np.where(df["fraction_zero"] > 0.80, "#d89c2b", "#2f8f5b"))
    ax.barh(df.index, df["fraction_zero"], color=colors)
    ax.axvline(0.95, color="#c93f3f", linestyle="--", linewidth=1.2, label="sparsity threshold = 0.95")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of candidate states where feature is zero")
    ax.set_title("Maintenance Feature Sparsity")
    ax.legend(loc="lower right")
    _save(fig, output_dir, "05_fraction_zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot maintenance feature-screening diagnostics.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="models/maintenance_feature_screening_greedy_5ep_14d",
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
