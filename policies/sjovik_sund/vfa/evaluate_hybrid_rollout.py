#!/usr/bin/env python3
"""
evaluate_hybrid_rollout.py

This script loads a fully trained offline Linear VFA model and wraps it 
inside a Hybrid Rollout Policy. It then runs a full simulation using your 
existing evaluation architecture to measure its final, real-world performance.
"""

import os
import sys
import argparse
from pathlib import Path
import time

# --- BULLETPROOF PATHING ---
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

# Import your VFA and Hybrid policies
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.vfa.HybridRolloutPolicy import HybridRolloutPolicy

# Import the baseline policy for comparison
from policies.do_nothing_policy import DoNothing

# --- CLEVER REUSE: Import your exact simulation runner and config! ---
# This ensures we keep all your existing CSV logging and metric tracking.
from policies.sjovik_sund.run_simulation_ingvild import (
    SimulationConfig,
    test_policies
)

def run_evaluation(
    model_path: str,
    lookahead_minutes: float,
    num_scenarios: int,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int
):
    print("=" * 60)
    print("  HYBRID ROLLOUT EVALUATION")
    print("=" * 60)
    
    model_file = Path(model_path)
    if not model_file.exists():
        print(f" Error: Could not find the trained model at {model_file}")
        sys.exit(1)

    # --- EXPERIMENT FEATURE MAPPING ---
    EXPERIMENTS = {
        "1_Linear_Reactive": ["rebalancing_imbalance", "vehicle_functional_load"],
        "2_Non_Linear_Reactive": ["squared_starvation_penalty", "squared_congestion_penalty", "vehicle_functional_load"],
        "3_Anticipatory_Spatial_Base": ["squared_starvation_penalty", "squared_congestion_penalty", "anticipated_demand_shortfall", "vehicle_functional_load", "proximity_to_demand_gravity"],
        "4_Contextual_Interactions": ["squared_starvation_penalty", "squared_congestion_penalty", "anticipated_demand_shortfall", "delivery_potential", "pickup_potential", "starvation_gravity", "congestion_gravity"]
    }

    # Automatically figure out which features to use based on the folder path!
    exp_name = next((key for key in EXPERIMENTS.keys() if key in model_path), None)
    
    if exp_name:
        active_features = EXPERIMENTS[exp_name]
        print(f"--> Auto-detected experiment '{exp_name}'. Injecting {len(active_features)} features.")
    else:
        print(f"Error: Could not detect experiment name from path '{model_path}'.")
        print("Make sure the folder name matches your EXPERIMENTS dictionary!")
        sys.exit(1)

    # 1. Load the frozen VFA (passing the correct active features!)
    print(f"--> Loading trained VFA from: {model_file.name}")
    trained_vfa = LinearVFAPolicy.load(model_file, active_features=active_features)
    
    # 🚨 CRITICAL: Ensure the VFA is strictly in exploitation mode 🚨
    trained_vfa.learning_mode = False 
    
    # 2. Wrap it in the Rollout framework
    print(f"--> Initializing Hybrid Rollout (Horizon: {lookahead_minutes}m, Scenarios: {num_scenarios})")
    hybrid_policy = HybridRolloutPolicy(
        trained_vfa=trained_vfa,
        lookahead_minutes=lookahead_minutes,
        num_scenarios=num_scenarios
    )
    
    # 3. Setup the policy dictionary for the simulator
    # (You can easily add your XPilot benchmark here later!)
    policy_dict = {
        f"DoNothing_Baseline": DoNothing(),
        f"Hybrid_Rollout_H{int(lookahead_minutes)}_S{num_scenarios}": hybrid_policy,
        f"VFA_Only_Standalone": trained_vfa  # Compare hybrid against the raw VFA!
    }
    
    # 4. Configure the simulation environment
    config = SimulationConfig()
    list_of_seeds = list(range(start_seed, start_seed + episodes))
    
    print("\nStarting simulation runs...")
    
    # Run using your existing test_policies function!
    test_policies(
        list_of_seeds=list_of_seeds,
        policy_dict=policy_dict,
        num_vehicles=vehicles,
        duration=duration_hours,
        use_multiprocessing=False, # Keep False until fast_clone() is fully thread-safe
        instance_name=instance,
        config=config
    )
    
    print("\n Evaluation complete! Check your standard output folders for the CSVs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Rollout Policy.")
    
    # Required Argument: Which model are we testing?
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to the trained VFA .pkl file (e.g., models/ablation_study/4_Contextual/vfa_run0.pkl)")
    
    # Rollout Tuning
    parser.add_argument("--lookahead", type=float, default=60.0, 
                        help="Rollout horizon in simulation minutes (default: 60)")
    parser.add_argument("--scenarios", type=int, default=3, 
                        help="Number of Monte Carlo scenarios per action (default: 3)")
    
    # Standard Simulation Settings
    parser.add_argument("--episodes", type=int, default=1, 
                        help="Number of evaluation episodes/seeds to run (default: 1)")
    parser.add_argument("--seed", type=int, default=999, 
                        help="Starting random seed (default: 999)")
    parser.add_argument("--duration", type=int, default=24 * 5, 
                        help="Duration in hours (default: 120)")
    parser.add_argument("--instance", type=str, default="TD_W34_old", 
                        help="Simulator instance name")
    parser.add_argument("--vehicles", type=int, default=1, 
                        help="Number of service vehicles")

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        lookahead_minutes=args.lookahead,
        num_scenarios=args.scenarios,
        episodes=args.episodes,
        start_seed=args.seed,
        duration_hours=args.duration,
        instance=args.instance,
        vehicles=args.vehicles
    )