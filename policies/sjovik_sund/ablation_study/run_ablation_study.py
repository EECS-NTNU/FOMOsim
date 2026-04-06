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
    # Test 1: Can it learn anything with basic, linear snapshots?
    "1_Linear_Reactive": [
        "rebalancing_imbalance",
        "vehicle_functional_load"
    ],
    
    # Test 2: Does punishing severe imbalances (squaring) improve routing?
    # ( REMOVE 'rebalancing_imbalance' to avoid redundancy)
    "2_Non_Linear_Reactive": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "vehicle_functional_load"
    ],
    
    # Test 3: Add forward-looking demand and basic spatial anchors
    "3_Anticipatory_Spatial_Base": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "anticipated_demand_shortfall",
        "vehicle_functional_load",
        "proximity_to_demand_gravity"
    ],
    
    # Test 4: Do interaction terms fix the ambiguity of standalone vehicle states?
    # ( REMOVE the standalone vehicle/spatial features and replace them with potentials)
    "4_Contextual_Interactions": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "anticipated_demand_shortfall",
        # --- Upgraded Vehicle & Spatial Interactions ---
        "delivery_potential",  
        "pickup_potential",
        "starvation_gravity",
        "congestion_gravity"
    ],
    
    # Test 5: Can it manage the trade-off between rebalancing and degradation?
    #"5_Full_System_Maintenance": [
        #"squared_starvation_penalty",
        ##"squared_congestion_penalty",
        ##"anticipated_demand_shortfall",
        ##"delivery_potential",  
        #"pickup_potential",
        #"starvation_gravity",
        #"congestion_gravity",
        # --- Maintenance Features ---
        #"trailer_cannibalization",
       # "global_onsite_backlog",
        #"demand_weighted_depot_backlog",
        #"depot_pull"
    #]
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