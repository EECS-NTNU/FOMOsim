#!/usr/bin/env python3
"""
evaluate_hybrid_rollout.py

Loads a trained offline Linear VFA model, wraps it in a HybridRolloutPolicy,
and evaluates it against baselines using the standard simulation runner.

Model path convention (from run_ablation_study.py):
    models/ablation_study/{exp_name}_alpha_{alpha}/vfa_{exp_name}_seed{seed}.pkl

The experiment name and alpha are auto-detected from the path. You can also
pass --experiment and/or --alpha explicitly to override detection.
"""

import os
import sys
import re
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.vfa.HybridRolloutPolicy import HybridRolloutPolicy
from policies.do_nothing_policy import DoNothing
from policies.sjovik_sund.run_simulation_ingvild import SimulationConfig, test_policies

# Single source of truth — imported directly from the ablation study definition
from policies.sjovik_sund.ablation_study.run_ablation_study import EXPERIMENTS

# Folder naming pattern produced by run_ablation_study.py
FOLDER_PATTERN = re.compile(r"^(.+)_alpha_([\d.]+)$")


def detect_experiment(model_path: str):
    """
    Walk up the model path looking for a folder that matches {exp}_alpha_{alpha}.
    Returns (exp_name, alpha_str) or (None, None) if not found.
    """
    for part in Path(model_path).parts:
        m = FOLDER_PATTERN.match(part)
        if m:
            return m.group(1), m.group(2)
    return None, None


def run_evaluation(
    model_path: str,
    lookahead_minutes: float,
    num_scenarios: int,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
    exp_override: str = None,
    alpha_override: str = None,
):
    print("=" * 60)
    print("  HYBRID ROLLOUT EVALUATION")
    print("=" * 60)

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Could not find model at {model_file}")
        sys.exit(1)

    # --- Auto-detect experiment and alpha from path ---
    exp_name, alpha_str = detect_experiment(model_path)
    if exp_override:
        exp_name = exp_override
    if alpha_override:
        alpha_str = alpha_override

    if exp_name is None or exp_name not in EXPERIMENTS:
        print(f"Error: Could not detect a valid experiment name from path '{model_path}'.")
        print(f"       Valid experiments: {list(EXPERIMENTS.keys())}")
        print(f"       Use --experiment to override.")
        sys.exit(1)

    active_features = EXPERIMENTS[exp_name]
    print(f"--> Experiment : {exp_name}")
    print(f"--> Alpha      : {alpha_str if alpha_str else 'unknown'}")
    print(f"--> Features   : {active_features}")

    # --- Load frozen VFA ---
    print(f"--> Loading VFA from: {model_file.name}")
    trained_vfa = LinearVFAPolicy.load(model_file, active_features=active_features)
    trained_vfa.learning_mode = False

    # --- Wrap in rollout ---
    print(f"--> Initializing Hybrid Rollout (horizon={lookahead_minutes}m, scenarios={num_scenarios})")
    hybrid_policy = HybridRolloutPolicy(
        trained_vfa=trained_vfa,
        lookahead_minutes=lookahead_minutes,
        num_scenarios=num_scenarios,
    )

    policy_dict = {
        "DoNothing_Baseline": DoNothing(),
        f"Hybrid_Rollout_H{int(lookahead_minutes)}_S{num_scenarios}": hybrid_policy,
        "VFA_Only_Standalone": trained_vfa,
    }

    config = SimulationConfig()
    list_of_seeds = list(range(start_seed, start_seed + episodes))

    print("\nStarting simulation runs...")
    test_policies(
        list_of_seeds=list_of_seeds,
        policy_dict=policy_dict,
        num_vehicles=vehicles,
        duration=duration_hours,
        use_multiprocessing=False,
        instance_name=instance,
        config=config,
    )

    print("\nEvaluation complete! Check your standard output folders for the CSVs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid Rollout Policy against baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained VFA .pkl file "
             "(e.g. models/ablation_study/V3_Rollout_alpha_0.5/vfa_V3_Rollout_seed1000.pkl)",
    )

    # Rollout tuning
    parser.add_argument("--lookahead", type=float, default=60.0,
                        help="Rollout horizon in simulation minutes")
    parser.add_argument("--scenarios", type=int, default=3,
                        help="Number of Monte Carlo scenarios per action")

    # Simulation settings
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of evaluation episodes (seeds) to run")
    parser.add_argument("--seed", type=int, default=999,
                        help="Starting random seed")
    parser.add_argument("--duration", type=int, default=24 * 5,
                        help="Simulation duration in hours")
    parser.add_argument("--instance", type=str, default="TD_W34_old",
                        help="Simulator instance name")
    parser.add_argument("--vehicles", type=int, default=1,
                        help="Number of service vehicles")

    # Overrides for auto-detection
    parser.add_argument(
        "--experiment", type=str, default=None,
        metavar="NAME",
        help=f"Override experiment name (auto-detected from path). Choices: {list(EXPERIMENTS.keys())}",
    )
    parser.add_argument(
        "--alpha", type=str, default=None,
        help="Override alpha value (auto-detected from path, used for logging only)",
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        lookahead_minutes=args.lookahead,
        num_scenarios=args.scenarios,
        episodes=args.episodes,
        start_seed=args.seed,
        duration_hours=args.duration,
        instance=args.instance,
        vehicles=args.vehicles,
        exp_override=args.experiment,
        alpha_override=args.alpha,
    )
