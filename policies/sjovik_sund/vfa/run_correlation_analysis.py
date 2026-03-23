import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Force Python to see the FOMOsim root directory ---
WORKSPACE_ROOT = Path(__file__).parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.train_vfa import run_simulation
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.policy import Policy

class FeatureLoggingPolicy(Policy):
    """A wrapper policy that logs all 25 VFA features before taking an action."""
    def __init__(self, vfa_policy):
        super().__init__(maintenance_enabled=True)
        self.vfa_policy = vfa_policy
        self.feature_log = []

    def get_best_action(self, state, vehicle):
        # 1. Initialize VFA caches if it's the first step
        if not self.vfa_policy._initialized:
            self.vfa_policy._lazy_init(state)
        
        # 2. Compute the 3-hour expectations for the candidate features
        curr_d = state.day() % 7
        curr_h = state.hour() % 24
        N = len(self.vfa_policy._station_ids)
        
        for i, s_id in enumerate(self.vfa_policy._station_ids):
            station = state.stations[s_id]
            flow, rent, ret = 0.0, 0.0, 0.0
            for offset in range(3):
                h = (curr_h + offset) % 24
                d = (curr_d + (curr_h + offset) // 24) % 7
                arr = station.arrive_intensities[d][h] if getattr(station, 'arrive_intensities', None) else 0
                lev = station.leave_intensities[d][h] if getattr(station, 'leave_intensities', None) else 0
                flow += (arr - lev)
                rent += lev
                ret += arr

        # 3. Extract base inventories
        base_func, base_onsite, base_depot = self.vfa_policy._extract_inventories(state, vehicle)

        # 4. Extract the full 25-feature vector for the CURRENT state (no action applied)
        phi = self.vfa_policy.extract_features(
            state, vehicle,
            base_func, base_onsite, base_depot,
            delta_func=0, delta_depot_cargo=0, delta_onsite_repairs=0,
            next_station_id=None
        )
        
        # 5. Map the vector back to the feature names and log it
        features_dict = dict(zip(self.vfa_policy.FEATURE_NAMES, phi))
        self.feature_log.append(features_dict)
        
        # 6. Ask the VFA policy to take the action so it ACTUALLY repairs bikes!
        return self.vfa_policy.get_best_action(state, vehicle)


def run_analysis():
    print("Running simulation to collect feature samples...")
    
    # Use LinearVFAPolicy to drive the simulation! 
    # We set learning_mode=True and tau=5.0 so it uses Boltzmann exploration 
    # to visit diverse states and actively test out repairs.
    vfa = LinearVFAPolicy(learning_mode=True, tau=5.0)
    logging_policy = FeatureLoggingPolicy(vfa)

    # Run a 30-day simulation to get a massive dataset of actions
    run_simulation(seed=42, policy=logging_policy, duration=24 * 60, num_vehicles=1)

    print(f"Success! Collected {len(logging_policy.feature_log)} state samples.\n")

    # --- CALCULATE CORRELATIONS ---
    df = pd.DataFrame(logging_policy.feature_log)
    
    # Drop features that literally had 0 variance across all 30 days (avoids NaN errors in correlation)
    df = df.loc[:, (df != df.iloc[0]).any()] 
    
    corr_matrix = df.corr(method='pearson')

    # --- PRINT WARNINGS FOR HIGH CORRELATION ---
    print("--- HIGHLY CORRELATED FEATURES (|r| > 0.85) ---")
    found_high_corr = False
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r_value = corr_matrix.iloc[i, j]
            if pd.notna(r_value) and abs(r_value) > 0.85:
                feat_a = corr_matrix.columns[i]
                feat_b = corr_matrix.columns[j]
                print(f" WARNING: '{feat_a}' and '{feat_b}' are highly correlated (r = {r_value:.3f})")
                found_high_corr = True
                
    if not found_high_corr:
        print(" Excellent! No highly correlated features found. Your VFA design is mathematically stable.")

    # --- PLOT THE HEATMAP ---
    plt.figure(figsize=(22, 18))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                vmin=-1, vmax=1, square=True, linewidths=.5, annot_kws={"size": 8})
    
    plt.title("VFA Feature Pearson Correlation Matrix (Exploration Mode)", fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    plot_path = "feature_correlation_heatmap.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nHeatmap saved to '{plot_path}'")

if __name__ == "__main__":
    run_analysis()