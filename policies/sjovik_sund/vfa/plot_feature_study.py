import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.vfa_features import REBALANCING_CORRELATION_FEATURES

# Use LaTeX styling when available; otherwise fall back to Matplotlib mathtext.
if shutil.which("latex"):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{mlmodern}"
    })
else:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
    })


def _display_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with underscore-free labels for LaTeX/plot readability."""
    display_df = df.copy()
    display_df.index = display_df.index.astype(str).str.replace('_', ' ')
    display_df.columns = display_df.columns.astype(str).str.replace('_', ' ')
    return display_df


def _plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    out_path: str,
    title: str,
    custom_cmap,
    figsize=(14, 12),
) -> None:
    plt.figure(figsize=figsize)
    sns.heatmap(
        _display_labels(corr_df),
        annot=True,
        cmap=custom_cmap,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        vmin=-1.0,
        vmax=1.0,
        annot_kws={"fontweight": "bold"},
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    print(f"Saved correlation matrix to {out_path}")
    plt.close()


def main():
    out_dir = WORKSPACE_ROOT / "models/feature_study"
    corr_path = out_dir / "feature_correlations.csv"

    if not os.path.exists(corr_path):
        print(f"Could not find the correlation CSV file at {corr_path}.")
        return

    # Load the data
    corr_df = pd.read_csv(corr_path, index_col=0)

    # Custom palette from Colorpallette.png
    custom_cmap = LinearSegmentedColormap.from_list('custom_palette', ['#3758d8', '#ffffff', '#db3249'])

    # 1. Plot the Correlation Matrix
    corr_out = os.path.join(out_dir, "feature_correlation_matrix.svg")
    _plot_correlation_heatmap(
        corr_df,
        corr_out,
        "Feature Correlation Matrix",
        custom_cmap,
    )

    # 1b. Plot a focused correlation matrix for the core rebalancing/FIM features.
    missing_features = [
        feat for feat in REBALANCING_CORRELATION_FEATURES
        if feat not in corr_df.index or feat not in corr_df.columns
    ]
    selected_features = [
        feat for feat in REBALANCING_CORRELATION_FEATURES
        if feat in corr_df.index and feat in corr_df.columns
    ]
    if missing_features:
        print("Missing focused-correlation features:", ", ".join(missing_features))
    if len(selected_features) >= 2:
        focused_corr = corr_df.loc[selected_features, selected_features]
        focused_corr_out = os.path.join(out_dir, "feature_correlation_matrix_rebalancing_subset.svg")
        _plot_correlation_heatmap(
            focused_corr,
            focused_corr_out,
            "Focused Rebalancing Feature Correlations",
            custom_cmap,
            figsize=(12, 10),
        )
    else:
        print("Skipping focused rebalancing correlation matrix: fewer than two requested features found.")

if __name__ == "__main__":
    main()
