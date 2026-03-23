import os
import sys
from pathlib import Path

# --- NEW: Force Python to see the FOMOsim root directory ---
WORKSPACE_ROOT = Path(__file__).parents[2]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))
# -----------------------------------------------------------

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import your existing simulation and policy components
# (Make sure the file name below matches what you named your clean simulation runner!)
from policies.sjovik_sund.vfa.train_vfa import run_simulation, SimulationConfig
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy, EpisodeTrainingPolicy
from policies.greedy_policy import GreedyPolicy
from policies.sjovik_sund.mdp.reward import RewardCalculator, RewardConfig
# --- PARETO CONFIGURATION ---
# Format: "Label": (weight_starvation, weight_congestion)
# Note: Penalties must be negative!
WEIGHT_RATIOS = {
    "1:2 (Favors Congestion)": (-1.0, -2.0),
    "1:1 (Equal Weight)":      (-1.0, -1.0),
    "2:1 (Favors Starvation)": (-2.0, -1.0),
    "3:1 (Heavy Starvation)":  (-3.0, -1.0),
    "5:1 (Extreme Starvation)":(-5.0, -1.0),
}

# Keep training short for the test run, but raise this (e.g., 200) for your final thesis graph!
TRAIN_EPISODES = 10
EVAL_SEEDS = 2   # How many seeds to test the frozen model on
DEBUG_OUTPUT_DIR = WORKSPACE_ROOT / "policies" / "sjovik_sund" / "vfa" / "output"


def _sanitize_label(label):
    return (
        label.lower()
        .replace(" ", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _save_theta_debug(label, theta_start, theta_end, episode_theta_history):
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = _sanitize_label(label)

    theta_df = pd.DataFrame({
        "feature_idx": np.arange(len(theta_end)),
        "theta_start": theta_start,
        "theta_end": theta_end,
        "theta_delta": theta_end - theta_start,
        "abs_theta_delta": np.abs(theta_end - theta_start),
    })
    theta_df.to_csv(DEBUG_OUTPUT_DIR / f"pareto_theta_{safe_label}.csv", index=False)

    history_df = pd.DataFrame(episode_theta_history)
    history_df.to_csv(DEBUG_OUTPUT_DIR / f"pareto_theta_history_{safe_label}.csv", index=False)


def _print_theta_debug(label, theta_start, theta_end):
    delta = theta_end - theta_start
    print(f"\n  Theta Debug for {label}")
    print(f"    ||theta_start|| = {np.linalg.norm(theta_start):.6f}")
    print(f"    ||theta_end||   = {np.linalg.norm(theta_end):.6f}")
    print(f"    ||delta||       = {np.linalg.norm(delta):.6f}")
    print(f"    max|delta|      = {np.max(np.abs(delta)):.6f}")
    print(f"    theta_start     = {np.array2string(theta_start, precision=4)}")
    print(f"    theta_end       = {np.array2string(theta_end, precision=4)}")
    print(f"    theta_delta     = {np.array2string(delta, precision=4)}")

def train_and_eval_ratio(label, w_starv, w_cong):
    print(f"\n{'='*50}")
    print(f"Testing Ratio: {label} (S:{w_starv}, C:{w_cong})")
    print(f"{'='*50}")
    
    # 1. Setup Custom Reward Calculator
    reward_config = RewardConfig(weight_starvation=w_starv, weight_congestion=w_cong)
    custom_reward_calc = RewardCalculator(config=reward_config)

    # 2. Initialize Policy
    vfa_policy = LinearVFAPolicy(
        alpha=0.01,
        gamma=0.99,
        tau=5.0,
        learning_mode=True,
        reward_calculator=custom_reward_calc
    )
    theta_start = vfa_policy.theta.copy()
    episode_theta_history = []
    
    greedy_policy = GreedyPolicy()
    
    # 3. Fast Training Loop (No CSV logging to save time)
    warmup_end_time = (7 * 60) + (4 * 24 * 60) # 4 days warmup
    tau_decay = (0.1 / 5.0) ** (1.0 / max(TRAIN_EPISODES - 1, 1))
    
    for ep in range(TRAIN_EPISODES):
        vfa_policy.set_temperature(5.0 * (tau_decay ** ep))
        
        episode_policy = EpisodeTrainingPolicy(
            vfa_policy=vfa_policy,
            greedy_policy=greedy_policy,
            warmup_end_time=warmup_end_time
        )
        
        # Run simulation silently
        run_simulation(seed=ep, policy=episode_policy, duration=24 * 14, num_vehicles=1)
        episode_theta_history.append({
            "episode": ep + 1,
            "tau": vfa_policy.tau,
            "theta_norm": float(np.linalg.norm(vfa_policy.theta)),
            "theta_mean": float(np.mean(vfa_policy.theta)),
            "theta_max_abs": float(np.max(np.abs(vfa_policy.theta))),
            "theta_0": float(vfa_policy.theta[0]) if len(vfa_policy.theta) > 0 else np.nan,
            "theta_1": float(vfa_policy.theta[1]) if len(vfa_policy.theta) > 1 else np.nan,
            "theta_2": float(vfa_policy.theta[2]) if len(vfa_policy.theta) > 2 else np.nan,
        })
        if (ep + 1) % 10 == 0:
            print(f"  Training Episode {ep+1}/{TRAIN_EPISODES} completed.")

    theta_end = vfa_policy.theta.copy()
    _print_theta_debug(label, theta_start, theta_end)
    _save_theta_debug(label, theta_start, theta_end, episode_theta_history)

    # 4. Frozen Evaluation Phase
    print("\n  Evaluating frozen policy...")
    vfa_policy.learning_mode = False # Freeze weights!
    total_s, total_c = 0, 0
    
    for eval_seed in range(1000, 1000 + EVAL_SEEDS):
        
        # --- FIX 1: Wrap the frozen VFA in the episode policy so it gets the exact same 4-day warmup! ---
        eval_policy = EpisodeTrainingPolicy(
            vfa_policy=vfa_policy,
            greedy_policy=greedy_policy,
            warmup_end_time=warmup_end_time
        )
        
        sim = run_simulation(seed=eval_seed, policy=eval_policy, duration=24 * 14, num_vehicles=1)

        # --- FIX 2: BULLETPROOF METRIC EXTRACTION (No overwriting!) ---
        s, c = 0, 0
        try:
            raw_metrics = sim.state.metrics.metrics
            # Check for either plural or singular metric names
            if "starvations" in raw_metrics and raw_metrics["starvations"]:
                s = raw_metrics["starvations"][-1][1]
            elif "starvation" in raw_metrics and raw_metrics["starvation"]:
                s = raw_metrics["starvation"][-1][1]
                
            if "long congestions" in raw_metrics and raw_metrics["long congestions"]:
                c = raw_metrics["long congestions"][-1][1]
            elif "long_congestion" in raw_metrics and raw_metrics["long_congestion"]:
                c = raw_metrics["long_congestion"][-1][1]
        except AttributeError:
            pass
        # -------------------------------------
        
        total_s += s
        total_c += c
        
    avg_s = total_s / EVAL_SEEDS
    avg_c = total_c / EVAL_SEEDS
    
    print(f"  Result -> Avg Starvations: {avg_s:.1f} | Avg Congestions: {avg_c:.1f}")
    return avg_s, avg_c


def run_analysis():
    results = {}
    theta_results = {}
    for label, (w_starv, w_cong) in WEIGHT_RATIOS.items():
        avg_s, avg_c = train_and_eval_ratio(label, w_starv, w_cong)
        results[label] = (avg_s, avg_c)
        theta_path_label = _sanitize_label(label)
        theta_df = pd.read_csv(DEBUG_OUTPUT_DIR / f"pareto_theta_{theta_path_label}.csv")
        theta_results[label] = theta_df["theta_end"].to_numpy()
        
    # --- PLOTTING THE PARETO FRONT ---
    plt.figure(figsize=(10, 7))
    
    x_vals, y_vals, labels = [], [], []
    for label, (s, c) in results.items():
        x_vals.append(c) # Congestions on X axis
        y_vals.append(s) # Starvations on Y axis
        labels.append(label)
        
    plt.scatter(x_vals, y_vals, color='blue', s=100)
    
    # Annotate the points
    for i, label in enumerate(labels):
        plt.annotate(label, (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Draw a line connecting them to show the curve
    sorted_indices = np.argsort(x_vals)
    plt.plot(np.array(x_vals)[sorted_indices], np.array(y_vals)[sorted_indices], 'r--', alpha=0.5)

    plt.title("Pareto Front: Starvation vs. Congestion Trade-off", fontsize=14)
    plt.xlabel("Average Total Congestions (Cost)", fontsize=12)
    plt.ylabel("Average Total Starvations (Cost)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save and show
    plt.savefig("pareto_analysis.png", dpi=300, bbox_inches='tight')
    print("\nAnalysis complete! Graph saved as 'pareto_analysis.png'.")

    if theta_results:
        print("\nFinal theta comparison across reward ratios:")
        labels_list = list(theta_results.keys())
        base_label = labels_list[0]
        base_theta = theta_results[base_label]
        for label in labels_list:
            theta = theta_results[label]
            diff_norm = np.linalg.norm(theta - base_theta)
            print(
                f"  {label:<30} "
                f"||theta||={np.linalg.norm(theta):.6f} "
                f"| diff vs {base_label} = {diff_norm:.6f}"
            )
    plt.show()

if __name__ == "__main__":
    run_analysis()
