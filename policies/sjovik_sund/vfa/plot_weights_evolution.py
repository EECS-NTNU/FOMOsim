#!/usr/bin/env python3
"""
plot_weights_evolution.py – Visualize VFA Weight Learning Curves

Reads the *_weights_evolution.csv file and generates comprehensive plots showing
how each feature weight (theta coefficient) evolves across training episodes.

Usage:
    python plot_weights_evolution.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- BULLETPROOF PATHING ---
WORKSPACE_ROOT = Path(__file__).resolve().parent
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

MODELS_DIR = WORKSPACE_ROOT / "models"


def plot_weights_evolution():
    """Load weights CSV and generate visualization plots."""
    
    if not MODELS_DIR.exists():
        print(f"Error: Directory does not exist -> {MODELS_DIR}")
        return

    # Find all *_weights_evolution.csv files
    csv_files = list(MODELS_DIR.glob("*_weights_evolution.csv"))
    if not csv_files:
        print(f"No *_weights_evolution.csv files found in {MODELS_DIR}")
        print("Make sure you have run train_vfa.py with the updated script!")
        return

    # Grab the most recently saved CSV file
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)

    print("=" * 72)
    print(" VFA WEIGHTS EVOLUTION ANALYSIS")
    print("=" * 72)
    print(f"Weights File: {latest_csv.name}")
    print(f"Location:     {latest_csv.parent}\n")

    # ---------------------------------------------------------
    # 1. Load the weights evolution CSV
    # ---------------------------------------------------------
    try:
        df = pd.read_csv(latest_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Extract episode and service level columns
    episodes = df["episode"].values
    service_levels = df["service_level"].values

    # Get all feature names (everything except 'episode' and 'service_level')
    feature_names = [col for col in df.columns if col not in ["episode", "service_level"]]
    n_features = len(feature_names)

    print(f"Total Training Episodes: {len(episodes)}")
    print(f"Number of Features:      {n_features}")
    print(f"Best Service Level:      {service_levels.max():.4f} (Episode {service_levels.argmax() + 1})")
    print(f"Final Service Level:     {service_levels[-1]:.4f}\n")

    # ---------------------------------------------------------
    # 2. Print weight summary statistics
    # ---------------------------------------------------------
    print("=" * 72)
    print(" WEIGHT STATISTICS ACROSS TRAINING")
    print("=" * 72)
    print(f"{'Feature':<35} {'Initial':>12} {'Final':>12} {'Change':>12}")
    print("-" * 72)

    for feature in feature_names:
        initial = df[feature].iloc[0]
        final = df[feature].iloc[-1]
        change = final - initial
        print(
            f"{feature:<35} {initial:>+12.6f} {final:>+12.6f} {change:>+12.6f}"
        )

    # ---------------------------------------------------------
    # 3. GENERATE PLOT 1: Individual weight evolution (subplots)
    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print(" GENERATING PLOTS")
    print("=" * 72)

    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    colors = plt.cm.tab20(np.linspace(0, 1, n_features))

    for idx, feature in enumerate(feature_names):
        ax = axes[idx]
        ax.plot(
            episodes,
            df[feature],
            marker="o",
            linewidth=2.5,
            markersize=5,
            color=colors[idx],
            alpha=0.8,
        )
        ax.fill_between(episodes, df[feature], alpha=0.2, color=colors[idx])

        # Add mean line
        mean_val = df[feature].mean()
        ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.5, alpha=0.6, label=f"Mean: {mean_val:.4f}")

        ax.set_title(f"{feature}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Weight Value (θ)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"VFA Feature Weight Evolution Across Training Episodes\n({latest_csv.name})",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    subplot_path = MODELS_DIR / latest_csv.name.replace("_weights_evolution.csv", "_weights_subplots.png")
    plt.savefig(subplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Weight Subplots -> {subplot_path.name}")

    # ---------------------------------------------------------
    # 4. GENERATE PLOT 2: All weights on single plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))

    for idx, feature in enumerate(feature_names):
        ax.plot(
            episodes,
            df[feature],
            marker="o",
            linewidth=2,
            markersize=4,
            label=feature,
            alpha=0.8,
        )

    ax.set_title(
        f"VFA Feature Weight Evolution (All Features)\n({latest_csv.name})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel("Weight Value (θ)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best", ncol=2)

    plt.tight_layout()

    combined_path = MODELS_DIR / latest_csv.name.replace("_weights_evolution.csv", "_weights_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Combined Weights Plot -> {combined_path.name}")

    # ---------------------------------------------------------
    # 5. GENERATE PLOT 3: Weight vs Service Level correlation
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    ax2 = ax.twinx()

    # Plot service level on secondary axis
    ax2.plot(
        episodes,
        service_levels,
        color="green",
        linewidth=3,
        label="Service Level",
        alpha=0.7,
        marker="s",
        markersize=6,
    )

    # Plot weights on primary axis
    for idx, feature in enumerate(feature_names):
        ax.plot(
            episodes,
            df[feature],
            linewidth=1.5,
            label=feature,
            alpha=0.6,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel("Weight Value (θ)", fontsize=12, color="black")
    ax2.set_ylabel("Service Level", fontsize=12, color="green")
    ax.set_title(
        f"Weight Evolution vs Service Level Progress\n({latest_csv.name})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax2.legend(fontsize=10, loc="upper right")

    plt.tight_layout()

    correlation_path = MODELS_DIR / latest_csv.name.replace("_weights_evolution.csv", "_weights_vs_service_level.png")
    plt.savefig(correlation_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Weight vs Service Level Plot -> {correlation_path.name}")

    # ---------------------------------------------------------
    # 6. GENERATE PLOT 4: Heatmap of weight evolution
    # ---------------------------------------------------------
    try:
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(14, 6))

        # Create matrix: rows = features, columns = episodes
        weight_matrix = df[feature_names].T.values

        sns.heatmap(
            weight_matrix,
            xticklabels=episodes,
            yticklabels=feature_names,
            cmap="RdYlBu_r",
            center=0,
            cbar_kws={"label": "Weight Value (θ)"},
            ax=ax,
        )

        ax.set_title(
            f"VFA Weight Heatmap (Features × Episodes)\n({latest_csv.name})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Training Episode", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)

        plt.tight_layout()

        heatmap_path = MODELS_DIR / latest_csv.name.replace("_weights_evolution.csv", "_weights_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved Weight Heatmap -> {heatmap_path.name}")

    except ImportError:
        print("⚠️  Seaborn not available; skipping heatmap visualization")

    # ---------------------------------------------------------
    # 7. Export statistics to CSV
    # ---------------------------------------------------------
    stats_data = {
        "Feature": feature_names,
        "Initial": [df[f].iloc[0] for f in feature_names],
        "Final": [df[f].iloc[-1] for f in feature_names],
        "Mean": [df[f].mean() for f in feature_names],
        "Std Dev": [df[f].std() for f in feature_names],
        "Min": [df[f].min() for f in feature_names],
        "Max": [df[f].max() for f in feature_names],
        "Change": [df[f].iloc[-1] - df[f].iloc[0] for f in feature_names],
    }

    stats_df = pd.DataFrame(stats_data)
    stats_path = MODELS_DIR / latest_csv.name.replace("_weights_evolution.csv", "_weights_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"✅ Exported Weight Statistics -> {stats_path.name}")

    print("\n" + "=" * 72)
    print(" ANALYSIS COMPLETE")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    plot_weights_evolution()
