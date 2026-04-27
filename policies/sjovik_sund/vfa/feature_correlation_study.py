#!/usr/bin/env python3
"""
feature_correlation_study.py

Runs a simulation with learning enabled to collect feature vectors and V(S^x)
estimates, then computes:
  - Descriptive statistics (min, max, mean, variance) per feature
  - Feature-feature correlation matrix
  - Feature-value Pearson r (bar chart + CSV)

Pillars 1 and 2 are enabled; Pillars 3 (maintenance) and 4 (logistics) are off.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.run_simulation_ingvild import run_simulation, SimulationConfig
from policies.sjovik_sund.vfa.train_vfa import INSTANCE_NAME, START_HOUR

class FeatureLoggingPolicy(LinearVFAPolicy):
    """Wraps LinearVFAPolicy to log every evaluated feature vector and V(S^x)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_log = []
        self.value_log   = []

    def extract_features(self, *args, **kwargs):
        phi = super().extract_features(*args, **kwargs)
        v   = float(np.dot(self.theta, phi))
        self.feature_log.append(phi.copy())
        self.value_log.append(v)
        return phi

def main():
    num_episodes = 5
    days_per_episode = 14

    print(f"Running {num_episodes} episodes × {days_per_episode}d with learning enabled...")
    policy = FeatureLoggingPolicy(
        maintenance_enabled=False,    # Pillar 3 off
        logistics_enabled=False,      # Pillar 4 off
        demand_horizon_enabled=True,  # Pillar 2 (FIM features) on
        learning_mode=True,
    )

    config = SimulationConfig()
    config.start_hour = START_HOUR

    for ep in range(num_episodes):
        run_simulation(
            seed=ep + 1,
            policy=policy,
            duration=24 * days_per_episode,
            num_vehicles=1,
            instance_name=INSTANCE_NAME,
            config=config,
        )
        policy.apply_batch_update()
        print(f"  Episode {ep + 1}/{num_episodes} done — {len(policy.feature_log)} samples so far")

    print(f"\nTotal feature samples collected: {len(policy.feature_log)}")

    df   = pd.DataFrame(policy.feature_log, columns=policy.FEATURE_NAMES)
    vals = np.array(policy.value_log)

    # ── Descriptive statistics ────────────────────────────────────────────────
    stats_df = df.describe().T[["mean", "std", "min", "max"]]
    stats_df["variance"] = df.var()
    stats_df = stats_df[["mean", "std", "variance", "min", "max"]]

    # ── Feature-feature correlation matrix ───────────────────────────────────
    corr_matrix = df.corr()

    # ── Feature-value Pearson r ───────────────────────────────────────────────
    pearson_r = pd.Series(
        {col: float(np.corrcoef(df[col].to_numpy(dtype=float), vals)[0, 1]) for col in df.columns},
        name="pearson_r_with_value",
    )

    # ── Save results ──────────────────────────────────────────────────────────
    output_dir = Path("models/feature_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path  = output_dir / "feature_statistics.csv"
    corr_path   = output_dir / "feature_correlations.csv"
    value_path  = output_dir / "feature_value_correlation.csv"
    plot_path   = output_dir / "feature_value_correlation.png"

    stats_df.to_csv(stats_path)
    corr_matrix.to_csv(corr_path)
    pearson_r.to_csv(value_path)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    sorted_r = pearson_r.sort_values()
    colors = ["green" if v >= 0 else "red" for v in sorted_r]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(sorted_r) + 1.5))
    ax.barh(sorted_r.index, sorted_r.to_numpy(dtype=float), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with V(S^x)")
    ax.set_title(f"Feature–Value Correlation  ({num_episodes} episodes × {days_per_episode}d)")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 50)
    print("FEATURE-VALUE PEARSON r")
    print("=" * 50)
    print(pearson_r.sort_values(ascending=False).to_string())

    print("\nResults saved to:")
    for p in [stats_path, corr_path, value_path, plot_path]:
        print(f"  {p}")

if __name__ == "__main__":
    main()
