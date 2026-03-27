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
    # 3. CALCULATE MOVING AVERAGES & SORT FEATURES
    # ---------------------------------------------------------
    # Smooth the jumpy service level using a Rolling Mean (Window = 10 episodes)
    window_size = min(10, len(episodes))
    sl_smoothed = pd.Series(service_levels).rolling(window=window_size, min_periods=1).mean()

    # Sort features by how "important" they became (absolute magnitude of final weight)
    # This separates the converging features from the noise!
    final_weights = {feat: abs(df[feat].iloc[-1]) for feat in feature_names}
    sorted_features = sorted(final_weights.keys(), key=lambda x: final_weights[x], reverse=True)

    # ---------------------------------------------------------
    # 4. GENERATE PLOT 2: All weights on single plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot EVERY feature with equal thickness and full labels
    for feature in sorted_features:
        ax.plot(
            episodes,
            df[feature],
            linewidth=2,
            alpha=0.8, 
            label=f"{feature} ({df[feature].iloc[-1]:+.2f})"
        )

    ax.set_title(
        f"VFA Feature Weight Evolution (All Features)\n({latest_csv.name})",
        fontsize=14, fontweight="bold",
    )
    # Move the legend outside the plot so it doesn't cover the lines
    ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5))

    # ---------------------------------------------------------
    # 5. GENERATE PLOT 3: Weight vs Smoothed Service Level
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))  # Made it a bit wider
    ax2 = ax.twinx()

    # Plot SMOOTHED service level
    ax2.plot(
        episodes, sl_smoothed,
        color="black", linewidth=4, label=f"Service Level ({window_size}-ep Avg)", alpha=0.9
    )

    # Plot ALL features 
    for feature in sorted_features:
        ax.plot(episodes, df[feature], linewidth=1.5, label=feature, alpha=0.7)

    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel("Weight Value (θ)", fontsize=12, color="black")
    ax2.set_ylabel("Service Level", fontsize=12, color="black")
    ax.set_title(
        f"All Feature Evolutions vs Smoothed Service Level\n({latest_csv.name})",
        fontsize=14, fontweight="bold",
    )
    
    # Put the weights legend on the left (outside) and Service Level on the right
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax2.legend(fontsize=10, loc="upper right")

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
