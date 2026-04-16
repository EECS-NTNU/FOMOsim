#!/usr/bin/env python3
"""
evaluate_hybrid_rollout.py  —  Evaluate trained VFA models inside the Hybrid Rollout Policy

Two modes of operation:

  1. BATCH MODE (mirrors the ablation study runner):
     Scans models/ablation_study/ for all saved .pkl files and evaluates each one.
     Loops over the same experiments and seeds as run_ablation_study.py.

       # All experiments, all seeds:
       python evaluate_hybrid_rollout.py

       # Specific experiments only:
       python evaluate_hybrid_rollout.py --experiments V3_Rollout LongTerm

       # Specific seeds only:
       python evaluate_hybrid_rollout.py --seeds 1000 2000

  2. SINGLE MODE:
     Point directly at one .pkl file. Features are auto-detected from the folder name,
     or overridden manually with --features.

       python evaluate_hybrid_rollout.py --model models/ablation_study/V3_Rollout/vfa_V3_Rollout_seed1000.pkl
       python evaluate_hybrid_rollout.py --model my_model.pkl --features squared_starvation_penalty work_ratio

Each evaluated model is compared against DoNothing and VFA-only baselines.
Results are written to the standard simulation_results/csv output folder.
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

# Single source of truth for experiment definitions
from policies.sjovik_sund.ablation_study.run_ablation_study import EXPERIMENTS

ABLATION_DIR = Path("models/ablation_study/SGDMINIBATCH_2/LongTerm_alpha_0.01_20260415_171622")


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation — one model file
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_file: Path,
    active_features: list[str],
    exp_name: str,
    seed: int,
    lookahead_minutes: float,
    num_scenarios: int,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
):
    print(f"\n{'='*60}")
    print(f"  {exp_name}  |  seed={seed}  |  {len(active_features)} features")
    print(f"  Model: {model_file.name}")
    print(f"{'='*60}")

    trained_vfa = LinearVFAPolicy.load(model_file, active_features=active_features)
    trained_vfa.learning_mode = False   # strictly exploitation — no TD updates

    hybrid_policy = HybridRolloutPolicy(
        trained_vfa=trained_vfa,
        lookahead_minutes=lookahead_minutes,
        num_scenarios=num_scenarios,
    )

    alpha_match = re.search(r'alpha_([\d.]+)', str(model_file))
    alpha_str = f"_A{alpha_match.group(1)}" if alpha_match else ""

    policy_dict = {
        f"{exp_name}_seed{seed}{alpha_str}_VFA_Only": trained_vfa,
        # Uncomment to also/instead run the Hybrid Rollout Policy
        f"{exp_name}_seed{seed}{alpha_str}_Hybrid_H{int(lookahead_minutes)}_S{num_scenarios}": hybrid_policy,
    }

    test_policies(
        list_of_seeds=list(range(start_seed, start_seed + episodes)),
        policy_dict=policy_dict,
        num_vehicles=vehicles,
        duration=duration_hours,
        use_multiprocessing=False,
        instance_name=instance,
        config=SimulationConfig(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch mode — loop over ablation study experiments and seeds
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    lookahead_minutes: float,
    num_scenarios: int,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
):
    print(f"\n{'='*60}")
    print("  RUNNING BASELINE: DoNothing")
    print(f"{'='*60}")
    test_policies(
        list_of_seeds=list(range(start_seed, start_seed + episodes)),
        policy_dict={"DoNothing_Baseline": DoNothing()},
        num_vehicles=vehicles,
        duration=duration_hours,
        use_multiprocessing=False,
        instance_name=instance,
        config=SimulationConfig(),
    )

    pkl_files = list(ABLATION_DIR.rglob("*.pkl"))
    
    if not pkl_files:
        print(f"Error: No .pkl models found in {ABLATION_DIR}")
        return

    print(f"\nBatch evaluation: Found {len(pkl_files)} model(s) to evaluate.")

    for i, model_file in enumerate(pkl_files):
        # Infer experiment properties
        try:
            active_features, exp_name = _detect_features(str(model_file))
        except ValueError as e:
            print(f"\n[{i+1}/{len(pkl_files)}] SKIP: {e}")
            continue
        
        # Extract seed if possible
        seed_match = re.search(r"seed(\d+)", model_file.name)
        seed_val = int(seed_match.group(1)) if seed_match else 0

        print(f"\n[{i+1}/{len(pkl_files)}] EVAL  {exp_name} (from: {model_file.name})")
        evaluate_model(
            model_file=model_file,
            active_features=active_features,
            exp_name=exp_name,
            seed=seed_val,
            lookahead_minutes=lookahead_minutes,
            num_scenarios=num_scenarios,
            episodes=episodes,
            start_seed=start_seed,
            duration_hours=duration_hours,
            instance=instance,
            vehicles=vehicles,
        )

    print(f"\n{'='*60}")
    print(f"Batch complete. Evaluated {len(pkl_files)} model(s).")
    print(f"Results written to simulation_results/csv/")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Single model mode
# ─────────────────────────────────────────────────────────────────────────────

def _detect_features(model_path: str) -> tuple[list[str], str]:
    """Infer feature set by matching the model path against known experiment names."""
    for exp_name, features in EXPERIMENTS.items():
        if exp_name in model_path:
            return features, exp_name
    raise ValueError(
        f"Could not detect an experiment name in path '{model_path}'.\n"
        f"Known experiments: {list(EXPERIMENTS.keys())}\n"
        f"Use --features to specify features manually."
    )


def run_single(
    model_path: str,
    features_override: list[str] | None,
    lookahead_minutes: float,
    num_scenarios: int,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
):
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: model not found at {model_file}")
        sys.exit(1)

    if features_override:
        active_features = features_override
        exp_name = model_file.stem
    else:
        active_features, exp_name = _detect_features(model_path)

    evaluate_model(
        model_file=model_file,
        active_features=active_features,
        exp_name=exp_name,
        seed=0,
        lookahead_minutes=lookahead_minutes,
        num_scenarios=num_scenarios,
        episodes=episodes,
        start_seed=start_seed,
        duration_hours=duration_hours,
        instance=instance,
        vehicles=vehicles,
    )
    print("\nEvaluation complete. Check simulation_results/csv/ for output.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained VFA models inside the Hybrid Rollout Policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Known experiments: {', '.join(EXPERIMENTS.keys())}",
    )

    # Mode selection
    mode = parser.add_argument_group("Mode (mutually exclusive — omit --model for batch mode)")
    mode.add_argument(
        "--model", type=str, default=None,
        help="Path to a single .pkl file (activates single mode)",
    )
    mode.add_argument(
        "--features", nargs="+", type=str, default=None, metavar="FEATURE",
        help="Override feature set manually (single mode only)",
    )

    # Batch mode options
    batch = parser.add_argument_group("Batch mode options (ignored in single mode)")
    batch.add_argument(
        "--experiments", nargs="+", type=str, default=None, metavar="NAME",
        help="Experiments to evaluate (default: all)",
    )
    batch.add_argument(
        "--seeds", nargs="+", type=int, default=[1000, 2000, 3000],
        help="Seeds to evaluate (default: 1000 2000 3000)",
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
    rollout = parser.add_argument_group("Rollout parameters")
    rollout.add_argument("--lookahead", type=float, default=60.0,
                         help="Rollout horizon in simulation minutes (default: 60)")
    rollout.add_argument("--scenarios", type=int, default=5,
                         help="Monte Carlo scenarios per action (default: 5)")

    # Simulation settings
    sim = parser.add_argument_group("Simulation settings")
    sim.add_argument("--episodes", type=int, default=1,
                     help="Evaluation episodes per model (default: 1)")
    sim.add_argument("--seed", type=int, default=9000,
                     help="Starting evaluation seed (default: 9000, kept separate from training seeds)")
    sim.add_argument("--duration", type=int, default=24 * 14,
                     help="Simulation duration in hours (default: 336)")
    sim.add_argument("--instance", type=str, default="TD_W34_old",
                     help="Simulator instance name")
    sim.add_argument("--vehicles", type=int, default=1,
                     help="Number of service vehicles")

    args = parser.parse_args()

    shared = dict(
        lookahead_minutes=args.lookahead,
        num_scenarios=args.scenarios,
        episodes=args.episodes,
        start_seed=args.seed,
        duration_hours=args.duration,
        instance=args.instance,
        vehicles=args.vehicles,
    )

    if args.model:
        run_single(model_path=args.model, features_override=args.features, **shared)
    else:
        run_batch(**shared)
