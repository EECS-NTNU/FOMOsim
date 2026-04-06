#!/usr/bin/env python3
"""
run_sensitivity_analysis.py
Automates a grid search over Gamma and Boltzmann Temperature parameters.
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import itertools
import sys

# --- Define the Hyperparameter Grid ---
gammas = [0.90, 0.95, 0.99]
tau_starts = [0.1, 0.5, 1.0]
tau_ends = [0.001, 0.01, 0.05]

# Baseline configuration parameters
EPISODES = 200
INSTANCE = "TD_W34_old" # Using the baseline instance
SEED = 3000

def run_grid_search():
    # Create a master directory for this grid search run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    master_dir = Path(f"models/grid_search_{timestamp}")
    master_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    combinations = list(itertools.product(gammas, tau_starts, tau_ends))
    total_runs = len(combinations)
    
    print(f"Starting Sensitivity Analysis: {total_runs} total configurations.")
    print("=" * 60)
    
    for i, (gamma, t_start, t_end) in enumerate(combinations, 1):
        print(f"\n[{i}/{total_runs}] Running config: Gamma={gamma}, Tau_Start={t_start}, Tau_End={t_end}")
        
        # Create a unique filename for this specific run
        run_name = f"vfa_G{gamma}_Ts{t_start}_Te{t_end}.pkl"
        save_path = master_dir / run_name
        
        # Build the command (Update the path to train_vfa.py here!)
        cmd = [
            sys.executable, "policies/sjovik_sund/vfa/train_vfa.py", # <-- UPDATED PATH
            "--episodes", str(EPISODES),
            "--instance", INSTANCE,
            "--seed", str(SEED),
            "--save", str(save_path),
            "--gamma", str(gamma),
            "--tau_start", str(t_start),
            "--tau_end", str(t_end)
        ]
        
        # Execute the training run
        try:
            # Using check=True will raise an exception if the script fails
            subprocess.run(cmd, check=True)
            print(f"Successfully completed run {i}/{total_runs}.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Run {i} failed with exit code {e.returncode}.")
            print("Moving to next configuration...")

    print("\n" + "=" * 60)
    print(f"Grid Search Complete! All models and learning curves saved to {master_dir}")

if __name__ == "__main__":
    run_grid_search()