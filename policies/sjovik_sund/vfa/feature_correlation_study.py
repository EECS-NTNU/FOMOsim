#!/usr/bin/env python3
"""
feature_correlation_study.py

Runs a simulation to collect feature vectors and computes descriptive
statistics (min, max, mean, variance) and a correlation matrix.
Categories A, C, and D are enabled. Category B (maintenance) is ignored.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.run_simulation_ingvild import run_simulation, SimulationConfig
from policies.sjovik_sund.vfa.train_vfa import INSTANCE_NAME, START_HOUR

class FeatureLoggingPolicy(LinearVFAPolicy):
    """Wraps LinearVFAPolicy to silently log every evaluated feature vector."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_log = []

    def extract_features(self, *args, **kwargs):
        phi = super().extract_features(*args, **kwargs)
        # Log only actual decision evaluations, not lookaheads.
        # This could include rollout evaluations, but provides a rich dataset.
        self.feature_log.append(phi)
        return phi

def main():
    print("Initializing tracking policy with Categories A, C, and D enabled...")
    policy = FeatureLoggingPolicy(
        maintenance_enabled=False,   # Ignore Category B
        shift_timing_enabled=False,   # Enable Category C
        temporal_enabled=True,       # Enable Category D
        learning_mode=False          # We don't need to update weights
    )

    # 1. Run a simulation to gather data
    print("Running simulation to collect feature data... (this might take a minute)")
    config = SimulationConfig()
    config.start_hour = START_HOUR
    
    run_simulation(
        seed=1,
        policy=policy,
        duration=24 * 7,  # Adjust days if you want a larger/smaller dataset
        num_vehicles=1,
        instance_name=INSTANCE_NAME,
        config=config
    )
    
    # 2. Convert logged features into a DataFrame
    print(f"Collected {len(policy.feature_log)} feature samples.")
    df = pd.DataFrame(policy.feature_log, columns=policy.FEATURE_NAMES)
    
    # 3. Calculate Key Metrics (Min, Max, Mean, Std, Variance)
    stats_df = df.describe().T[['mean', 'std', 'min', 'max']]
    stats_df['variance'] = df.var()
    stats_df = stats_df[['mean', 'std', 'variance', 'min', 'max']] # Reorder

    # 4. Calculate Correlation Matrix
    corr_matrix = df.corr()

    # 5. Save results to CSV
    output_dir = Path("models/feature_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats_path = output_dir / "feature_statistics.csv"
    corr_path = output_dir / "feature_correlations.csv"
    
    stats_df.to_csv(stats_path)
    corr_matrix.to_csv(corr_path)

    print("\n" + "="*50)
    print("FEATURE STATISTICS HIGHLIGHTS")
    print("=" * 50)
    print(stats_df.head(10).to_string()) # Print top 10 for preview
    
    print("\nResults saved to:")
    print(f" - {stats_path}")
    print(f" - {corr_path}")
    print("\nTip: Look for feature pairs in 'feature_correlations.csv' with a correlation > 0.8 or < -0.8 to identify redundancies!")

if __name__ == "__main__":
    main()