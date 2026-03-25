import os
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.train_vfa import train

# Define your experimental subsets here!
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

def run_all_experiments(episodes: int = 100):
    base_dir = Path("models/ablation_study")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Define 3 different starting offsets for our multiple runs
    MACRO_SEEDS = [1000] 
    
    for exp_name, features in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"STARTING EXPERIMENT: {exp_name}")
        print(f"Features: {features}")
        print(f"{'='*60}")
        
        exp_dir = base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Loop through each macro-seed and train from scratch
        for run_id, seed_offset in enumerate(MACRO_SEEDS):
            print(f"  --> Run {run_id + 1}/{len(MACRO_SEEDS)} (Seed Offset: {seed_offset})")
            
            # Save the model and logs with the run_id in the name
            save_path = exp_dir / f"vfa_{exp_name}_run{run_id}.pkl"
            
            train(
                num_episodes=episodes,
                save_path=save_path,
                seed_offset=seed_offset, # Pass the macro-seed here!
                active_features=features
            )

if __name__ == "__main__":
    run_all_experiments(episodes=2)
