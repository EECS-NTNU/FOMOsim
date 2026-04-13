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


def resolve_model_path(exp_name: str, alpha: str, seed: int) -> Path:
    """Build the canonical model path from experiment + alpha + seed."""
    folder = WORKSPACE_ROOT / "models" / "ablation_study" / f"{exp_name}_alpha_{alpha}"
    return folder / f"vfa_{exp_name}_seed{seed}.pkl"


def run_evaluation(
    model_path: str | None,
    lookahead_minutes: float,
    num_scenarios: int,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
    exp_name: str = None,
    alpha: str = None,
    seed: int = 1000,
):
    print("=" * 60)
    print("  HYBRID ROLLOUT EVALUATION")
    print("=" * 60)

    # --- Resolve experiment and model file ---
    if model_path:
        # Explicit path: auto-detect exp/alpha from it, allow CLI overrides
        detected_exp, detected_alpha = detect_experiment(model_path)
        exp_name  = exp_name  or detected_exp
        alpha     = alpha     or detected_alpha
        model_file = Path(model_path)
    elif exp_name and alpha:
        # Build path from experiment + alpha + seed
        model_file = resolve_model_path(exp_name, alpha, seed)
    else:
        print("Error: Provide either --model or both --experiment and --alpha.")
        sys.exit(1)

    if exp_name is None or exp_name not in EXPERIMENTS:
        print(f"Error: Could not resolve a valid experiment name.")
        print(f"       Valid experiments: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    if not model_file.exists():
        print(f"Error: Could not find model at {model_file}")
        sys.exit(1)

    active_features = EXPERIMENTS[exp_name]
    print(f"--> Experiment : {exp_name}")
    print(f"--> Alpha      : {alpha if alpha else 'unknown'}")
    print(f"--> Model      : {model_file}")
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
        "--model", type=str, default=None,
        help="Path to trained VFA .pkl file. "
             "If omitted, path is built from --experiment, --alpha, and --model_seed. "
             "(e.g. models/ablation_study/V3_Rollout_alpha_0.5/vfa_V3_Rollout_seed1000.pkl)",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        metavar="NAME",
        help=f"Experiment name. Required when --model is not given. Choices: {list(EXPERIMENTS.keys())}",
    )
    parser.add_argument(
        "--alpha", type=str, default=None,
        help="Alpha value (e.g. 0.5). Required when --model is not given.",
    )
    parser.add_argument(
        "--model_seed", type=int, default=1000,
        help="Seed of the trained model to load when building path from --experiment/--alpha",
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
                        help="Starting random seed for evaluation")
    parser.add_argument("--duration", type=int, default=24 * 5,
                        help="Simulation duration in hours")
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
        vehicles=args.vehicles,
        exp_name=args.experiment,
        alpha=args.alpha,
        seed=args.model_seed,
    )
