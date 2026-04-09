import os
import sys
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.train_vfa import train

# Define your experimental subsets here!
EXPERIMENTS = {
    # --- AXIS 1: Temporal Horizon Depth (controls how far ahead the VFA "sees") ---
    
    # Horizon-0: Pure reactive snapshot (no temporal info at all)
    "H0_Reactive_Snapshot": [
        "rebalancing_imbalance",
        "vehicle_functional_load",
    ],
    
    # Horizon-1: One-step demand anticipation
    "H1_One_Step_Demand": [
        "squared_starvation_penalty",   # uses time-indexed target
        "squared_congestion_penalty",
        "anticipated_demand_shortfall", # one-step activity
        "vehicle_functional_load",
    ],
    
    # Horizon-N: Multi-step demand integration
    "HN_Multi_Horizon": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "multi_horizon_starvation_risk",  # NEW: integrate over H hours
        "time_of_day_fraction",           # NEW: φ_T1
        "hours_until_peak_fraction",      # NEW: φ_T3
        "vehicle_functional_load",
    ],
    
    # --- AXIS 2: Spatial recoverability ---
    "Spatial_Recoverability": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "imbalance_weighted_distance",    # NEW: φ_R1
        "starvation_severity_max",        # NEW: φ_R2
        "delivery_potential",
        "pickup_potential",
    ],
    
    # --- AXIS 3: Full candidate ---
    "Full_VFA_Candidate": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "multi_horizon_starvation_risk",
        "time_of_day_fraction",
        "delivery_potential",
        "pickup_potential",
        "starvation_gravity",
        "congestion_gravity",
        "imbalance_weighted_distance",
    ],
}

def run_all_experiments(seeds: list[int], episodes: int = 200):
    base_dir = Path("models/ablation_study")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for exp_name, features in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"STARTING EXPERIMENT: {exp_name}")
        print(f"Features: {features}")
        print(f"{'='*60}")
        
        exp_dir = base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Loop through each macro-seed provided via the terminal
        for run_id, seed_offset in enumerate(seeds):
            print(f"  --> Run {run_id + 1}/{len(seeds)} (Seed Offset: {seed_offset})")
            
            # Save the model and logs with the specific SEED in the name
            save_path = exp_dir / f"vfa_{exp_name}_seed{seed_offset}.pkl"
            
            train(
                num_episodes=episodes,
                save_path=save_path,
                seed_offset=seed_offset, 
                active_features=features
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VFA Feature Ablation Study")
    
    # Allow passing multiple seeds directly from the terminal
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1000, 2000, 3000],
        help="List of seed offsets to run for robust averaging (e.g., --seeds 1000 2000 3000)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of training episodes per run"
    )
    
    args = parser.parse_args()

    # Pass the parsed arguments into the runner
    run_all_experiments(seeds=args.seeds, episodes=args.episodes)