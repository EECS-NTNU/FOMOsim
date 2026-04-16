#!/usr/bin/env python3
"""
evaluate_nn_rollout.py  —  Evaluate trained NN models inside the NNRolloutPolicy

Two modes:

  1. BATCH MODE:
     Scans NN/models/ for all .pt files and evaluates each one.

       python evaluate_nn_rollout.py

       # Specific model files only:
       python evaluate_nn_rollout.py --models models/nn_model_final_seed0.pt

  2. SINGLE MODE:
     Point directly at one .pt file.

       python evaluate_nn_rollout.py --model models/nn_model_final_seed0.pt

Each evaluated model is compared against DoNothing and pure-NN (no rollout) baselines.
Results are written to simulation_results/csv/.
"""

import os
import sys
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.NN.train_nn_rollout import load_nn_model
from policies.sjovik_sund.NN.NNRolloutPolicy import NNRolloutPolicy
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.do_nothing_policy import DoNothing
from policies.sjovik_sund.run_simulation_ingvild import SimulationConfig, test_policies
from settings import ENABLE_COMPONENT_FAILURES

NN_MODELS_DIR = Path(__file__).parent / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation — one .pt model file
# ─────────────────────────────────────────────────────────────────────────────

def _make_candidate_vfa() -> LinearVFAPolicy:
    """
    Build a default LinearVFAPolicy used only for candidate generation and
    reward config inside NNRolloutPolicy.  Weights are never queried.
    """
    vfa = LinearVFAPolicy(
        learning_mode=False,
        maintenance_enabled=ENABLE_COMPONENT_FAILURES,
    )
    vfa.learning_mode = False
    return vfa


def evaluate_model(
    model_file: Path,
    lookahead_minutes: float,
    num_scenarios: int,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
    depot_id: str = None,
):
    model_name = model_file.stem

    print(f"\n{'='*60}")
    print(f"  Model: {model_file.name}")
    print(f"  Lookahead: {lookahead_minutes} min  |  Scenarios: {num_scenarios}")
    print(f"{'='*60}")

    nn_model = load_nn_model(str(model_file))

    candidate_vfa = _make_candidate_vfa()

    nn_rollout = NNRolloutPolicy(
        nn_model=nn_model,
        candidate_vfa=candidate_vfa,
        lookahead_minutes=lookahead_minutes,
        num_scenarios=num_scenarios,
        maintenance_enabled=ENABLE_COMPONENT_FAILURES,
        depot_id=depot_id,
    )

    # Pure NN (no rollout) baseline — needs its own candidate_vfa instance
    nn_greedy_vfa = _make_candidate_vfa()
    nn_greedy = NNRolloutPolicy(
        nn_model=nn_model,
        candidate_vfa=nn_greedy_vfa,
        lookahead_minutes=0.0,   # no lookahead → pure NN greedy
        num_scenarios=1,
        maintenance_enabled=ENABLE_COMPONENT_FAILURES,
        depot_id=depot_id,
    )

    policy_dict = {
        "DoNothing":                                          DoNothing(),
        f"{model_name}_NNGreedy":                             nn_greedy,
        f"{model_name}_NNRollout_H{int(lookahead_minutes)}_S{num_scenarios}": nn_rollout,
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
# Batch mode — scan NN/models/ for all .pt files
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    lookahead_minutes: float,
    num_scenarios: int,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
    depot_id: str = None,
    pattern: str = "*.pt",
):
    model_files = sorted(NN_MODELS_DIR.glob(pattern))

    if not model_files:
        print(f"No .pt files found in {NN_MODELS_DIR}")
        return

    total   = len(model_files)
    skipped = 0
    print(f"\nBatch evaluation: {total} model(s) found in {NN_MODELS_DIR}")

    for i, model_file in enumerate(model_files, 1):
        print(f"\n[{i}/{total}] EVAL  {model_file.name}")
        try:
            evaluate_model(
                model_file=model_file,
                lookahead_minutes=lookahead_minutes,
                num_scenarios=num_scenarios,
                episodes=episodes,
                start_seed=start_seed,
                duration_hours=duration_hours,
                instance=instance,
                vehicles=vehicles,
                depot_id=depot_id,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Batch complete. Evaluated: {total - skipped}/{total}  |  Errors: {skipped}")
    print(f"Results written to simulation_results/csv/")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained NN models inside NNRolloutPolicy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_argument_group("Mode (omit --model for batch mode)")
    mode.add_argument(
        "--model", type=str, default=None,
        help="Path to a single .pt file (activates single mode)",
    )
    mode.add_argument(
        "--pattern", type=str, default="*.pt",
        help="Glob pattern for batch mode (default: *.pt)",
    )

    rollout = parser.add_argument_group("Rollout parameters")
    rollout.add_argument("--lookahead", type=float, default=60.0,
                         help="Rollout horizon in simulation minutes (default: 60)")
    rollout.add_argument("--scenarios", type=int, default=3,
                         help="Monte Carlo scenarios per action (default: 3)")

    sim = parser.add_argument_group("Simulation settings")
    sim.add_argument("--episodes",  type=int,   default=5,
                     help="Evaluation episodes per model (default: 5)")
    sim.add_argument("--seed",      type=int,   default=9000,
                     help="Starting evaluation seed (default: 9000)")
    sim.add_argument("--duration",  type=int,   default=24 * 5,
                     help="Simulation duration in hours (default: 120)")
    sim.add_argument("--instance",  type=str,   default="TD_W34_old",
                     help="Simulator instance name")
    sim.add_argument("--vehicles",  type=int,   default=1,
                     help="Number of service vehicles")
    sim.add_argument("--depot_id",  type=str,   default=None,
                     help="Depot station ID (e.g. 'D0'). Auto-resolved if omitted.")

    args = parser.parse_args()

    shared = dict(
        lookahead_minutes=args.lookahead,
        num_scenarios=args.scenarios,
        episodes=args.episodes,
        start_seed=args.seed,
        duration_hours=args.duration,
        instance=args.instance,
        vehicles=args.vehicles,
        depot_id=args.depot_id,
    )

    if args.model:
        model_file = Path(args.model)
        if not model_file.exists():
            print(f"Error: model not found at {model_file}")
            sys.exit(1)
        evaluate_model(model_file=model_file, **shared)
        print("\nEvaluation complete. Check simulation_results/csv/ for output.")
    else:
        run_batch(pattern=args.pattern, **shared)
