#!/usr/bin/env python3
"""
evaluate_nn_rollout.py  —  Evaluate trained NN models inside the NNRolloutPolicy

Two modes:

  1. BATCH MODE:
     Scans NN/models/ for all .pt files and evaluates each one.
       python evaluate_nn_rollout.py

  2. SINGLE MODE:
     Point directly at one .pt file.
       python evaluate_nn_rollout.py --model models/nn_model_final_seed0.pt

By default, each evaluated model is tested as an NNRolloutPolicy, using the same
RunLogger-based testing path as evaluate_hybrid_rollout.py. Optional baselines
can be enabled with CLI flags.
"""

import os
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.NN.train_nn_rollout import load_nn_model
from policies.sjovik_sund.NN.NNRolloutPolicy import NNRolloutPolicy
from policies.sjovik_sund.NN.NNAnalyticalRolloutPolicy import NNAnalyticalRolloutPolicy
from policies.do_nothing_policy import DoNothing
from policies.sjovik_sund.run_simulation_ingvild import SimulationConfig, test_policies
from policies.sjovik_sund.vfa.run_logger import RunLogger

# --- NEW BASELINE IMPORTS ---
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.NN.NNGreedyPolicy import NNGreedyPolicy
from policies.sjovik_sund.mdp.mdp_config import MDPConfig
from policies.sjovik_sund.NN.rollout_debug_logger import PruningDebugLogger
from policies.greedy_policy_maintenance import GreedyMaintenancePolicy

NN_MODELS_DIR = Path(__file__).parent / "models"


def _parse_model_hyperparams(model_stem: str):
    """Extract lr, tau, freq from a model filename stem"""
    lr, tau, freq = None, None, None
    m_lr = re.search(r"lr([0-9.]+)", model_stem)
    if m_lr: lr = m_lr.group(1)
    
    m_tau = re.search(r"tau([0-9.]+)", model_stem)
    if m_tau: tau = m_tau.group(1)
    
    m_freq = re.search(r"freq([0-9]+)", model_stem)
    if m_freq: freq = m_freq.group(1)
    
    return lr, tau, freq


def evaluate_model(
    model_file: Path,
    lookahead_minutes: float,
    num_scenarios: int,
    n_rollout_candidates: int,
    congestion_weight: float,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
    depot_id: str = None,
    debug_log_every: int = 0,
    maintenance_enabled: bool = False,
    n_screening_scenarios: int = 3,
    n_survivors: int = 4,
    n_time_steps: int = 4,
    run_logger=None,
    write_csv: bool = False,
    run_rollout: bool = True,
    use_analytical_rollout: bool = True,
    run_donothing: bool = False,
    run_nn_greedy: bool = False,
    run_linear_vfa: bool = False,
    run_greedy_maintenance: bool = False,
):
    model_name = model_file.stem
    lr, tau, freq = _parse_model_hyperparams(model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build a compact hyper-param tag for the output filename
    hp_parts = []
    if tau  is not None: hp_parts.append(f"tau{tau}")
    if freq is not None: hp_parts.append(f"freq{freq}")
    if lr   is not None: hp_parts.append(f"lr{lr}")
    hp_str = ("_" + "_".join(hp_parts)) if hp_parts else ""

    eval_key = (
        f"{model_name}_{'NNAnalytical' if use_analytical_rollout else 'NNRollout'}"
        f"_H{int(lookahead_minutes)}_S{num_scenarios}"
        f"{hp_str}_{timestamp}"
    )

    print(f"\n{'='*60}")
    print(f"  Model: {model_file.name}")
    print(f"  Lookahead: {lookahead_minutes} min  |  Scenarios: {num_scenarios}")
    if hp_parts:
        print(f"  Hyperparams: {', '.join(hp_parts)}")
    print(f"  Eval timestamp: {timestamp}")
    print(f"{'='*60}")

    eval_seeds = list(range(start_seed, start_seed + episodes))

    def _test_one(policy_name: str, policy, policy_type: str, results_only: bool = False):
        if run_logger is not None:
            run_logger.set_run_label(
                model_name,
                0.0,
                policy_type=policy_type,
                duration_hours=duration_hours,
                num_vehicles=vehicles,
                instance_name=instance,
                results_only=results_only,
            )

        test_policies(
            list_of_seeds=eval_seeds,
            policy_dict={policy_name: policy},
            num_vehicles=vehicles,
            duration=duration_hours,
            use_multiprocessing=False,
            instance_name=instance,
            config=SimulationConfig(),
            run_logger=run_logger,
            write_csv=write_csv,
        )

    if run_donothing:
        print(f"\n{'='*60}")
        print("  RUNNING BASELINE: DoNothing")
        print(f"{'='*60}")
        _test_one("DoNothing_Baseline", DoNothing(), "do-nothing", results_only=True)

    if run_linear_vfa:
        print(f"\n{'='*60}")
        print("  RUNNING BASELINE: LinearVFA")
        print(f"{'='*60}")
        _test_one(
            "LinearVFA",
            LinearVFAPolicy(learning_mode=False, maintenance_enabled=maintenance_enabled),
            "vfa",
        )

    if run_greedy_maintenance:
        print(f"\n{'='*60}")
        print("  RUNNING BASELINE: GreedyMaintenance")
        print(f"{'='*60}")
        _test_one("GreedyMaintenance", GreedyMaintenancePolicy(), "greedy-maintenance")

    if not (run_rollout or run_nn_greedy):
        return

    nn_model = load_nn_model(str(model_file))

    pruning_logger = None
    if debug_log_every > 0:
        log_path = NN_MODELS_DIR.parent / "debug_logs" / f"pruning_{model_file.stem}_{timestamp}.log"
        pruning_logger = PruningDebugLogger(log_path=log_path, log_every=debug_log_every)
        print(f"  Debug log → {log_path}")

    mdp_config = MDPConfig.full_maintenance() if maintenance_enabled else MDPConfig.no_maintenance()

    if run_nn_greedy:
        print(f"\n{'='*60}")
        print("  RUNNING NN greedy baseline")
        print(f"{'='*60}")
        _test_one(
            f"{model_name}_NNGreedy",
            NNGreedyPolicy(nn_model=nn_model, config=mdp_config, depot_id=depot_id),
            "nn-greedy",
        )

    if run_rollout:
        if use_analytical_rollout:
            nn_rollout = NNAnalyticalRolloutPolicy(
                nn_model=nn_model,
                lookahead_minutes=lookahead_minutes,
                num_scenarios=num_scenarios,
                n_rollout_candidates=n_rollout_candidates,
                n_time_steps=n_time_steps,
                maintenance_enabled=maintenance_enabled,
                depot_id=depot_id,
                congestion_weight=congestion_weight,
            )
            policy_type = "nn-analytical"
        else:
            nn_rollout = NNRolloutPolicy(
                nn_model=nn_model,
                lookahead_minutes=lookahead_minutes,
                num_scenarios=num_scenarios,
                n_rollout_candidates=n_rollout_candidates,
                n_screening_scenarios=n_screening_scenarios,
                n_survivors=n_survivors,
                maintenance_enabled=maintenance_enabled,
                depot_id=depot_id,
                congestion_weight=congestion_weight,
                pruning_logger=pruning_logger,
            )
            policy_type = "nn-rollout"
        _test_one(eval_key, nn_rollout, policy_type)

    if pruning_logger is not None:
        pruning_logger.close()


# ─────────────────────────────────────────────────────────────────────────────
# Batch mode — scan NN/models/ for all .pt files
# ─────────────────────────────────────────────────────────────────────────────
def run_batch(pattern: str, log_decisions: bool = False, **kwargs):
    models = list(NN_MODELS_DIR.glob(pattern))
    if not models:
        print(f"No models found matching '{pattern}' in {NN_MODELS_DIR}")
        return

    run_logger = RunLogger(log_decisions=log_decisions)
    print(f"Found {len(models)} model(s) for batch evaluation.")
    try:
        for mf in sorted(models):
            evaluate_model(model_file=mf, run_logger=run_logger, **kwargs)
    finally:
        run_logger.close()
        print(f"\nRun logs written to: {run_logger.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained NN models inside NNRolloutPolicy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_argument_group("Mode (omit --model for batch mode)")
    mode.add_argument(
        "--model",
        type=str,
        default="policies/sjovik_sund/NN/models/nn_model_best_greedy_seed1000_arch_128-64-32_20260504_143352.pt",
        help="Path to a single .pt file (activates single mode)",
    )
    mode.add_argument(
        "--pattern", type=str, default="*.pt",
        help="Glob pattern for batch mode (default: *.pt)",
    )

    rollout = parser.add_argument_group("Rollout parameters")
    rollout.add_argument("--lookahead", type=float, default=60.0,
                         help="Rollout horizon in simulation minutes (default: 60)")
    rollout.add_argument("--scenarios", type=int, default=8,
                         help="Monte Carlo scenarios per action (default: 8)")
    # --- NEW ARGUMENT ---
    rollout.add_argument("--candidates", type=int, default=999,
                         help="Number of candidates to run full rollout on (default: 999)")
    
    # --- NEW ARGUMENT ---
    rollout.add_argument("--congestion_weight", type=float, default=-1.0,
                         help="Reward penalty for congestions. Default: -1.0. Use -1.0 for strict 1:1.")
    rollout.add_argument("--n_screening", type=int, default=3,
                         help="Stage-1 scenarios per candidate in two-stage screening (default: 3)")
    rollout.add_argument("--n_survivors", type=int, default=7,
                         help="Candidates advanced from stage-1 to stage-2 (default: 7)")
    rollout.add_argument("--n_time_steps", type=int, default=4,
                         help="Sub-intervals per analytical rollout horizon (default: 4)")

    debug = parser.add_argument_group("Debug / tracing")
    debug.add_argument("--debug", type=int, default=0, metavar="N",
                       help="Write pruning trace to debug_logs/. Log every Nth decision (0=off, 1=all, 10=every 10th)")
    debug.add_argument("--maintenance", action="store_true",
                       help="Include maintenance actions as candidates (default: off)")

    testing = parser.add_argument_group("Testing options")
    testing.add_argument("--log_decisions", action="store_true", default=False,
                         help="Write decisions.csv in run_logs/. Can be large for long runs.")
    testing.add_argument("--run_donothing", "--run_donoting", dest="run_donothing",
                         action="store_true", default=False,
                         help="Also run the DoNothing baseline.")
    testing.add_argument("--run_nn_greedy", action="store_true", default=False,
                         help="Also run the NN greedy baseline without rollout.")
    testing.add_argument("--run_linear_vfa", action="store_true", default=False,
                         help="Also run the default LinearVFA baseline.")
    testing.add_argument("--run_greedy_maintenance", action="store_true", default=False,
                         help="Also run the GreedyMaintenance baseline.")
    testing.add_argument("--no_rollout", action="store_true", default=False,
                         help="Skip NNRolloutPolicy evaluation, useful for baselines only.")
    testing.add_argument("--simulation_rollout", action="store_true", default=False,
                         help="Use old simulator-cloning NN rollout instead of fast analytical rollout.")

    sim = parser.add_argument_group("Simulation settings")
    sim.add_argument("--episodes",  type=int,   default=5,
                     help="Evaluation episodes per model (default: 5)")
    sim.add_argument("--seed",      type=int,   default=42,
                     help="Starting evaluation seed (default: 42)")
    sim.add_argument("--duration",  type=int,   default=24 * 7,
                     help="Simulation duration in hours (default: 336)")
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
        n_rollout_candidates=args.candidates,
        congestion_weight=args.congestion_weight,
        n_screening_scenarios=args.n_screening,
        n_survivors=args.n_survivors,
        n_time_steps=args.n_time_steps,
        episodes=args.episodes,
        start_seed=args.seed,
        duration_hours=args.duration,
        instance=args.instance,
        vehicles=args.vehicles,
        depot_id=args.depot_id,
        debug_log_every=args.debug,
        maintenance_enabled=args.maintenance,
        run_rollout=not args.no_rollout,
        use_analytical_rollout=not args.simulation_rollout,
        run_donothing=args.run_donothing,
        run_nn_greedy=args.run_nn_greedy,
        run_linear_vfa=args.run_linear_vfa,
        run_greedy_maintenance=args.run_greedy_maintenance,
    )

    if args.model:
        model_file = Path(args.model)
        if not model_file.exists():
            model_file = NN_MODELS_DIR / model_file  # try relative to models/
        if not model_file.exists():
            print(f"Error: model not found at {model_file}")
            sys.exit(1)
        run_logger = RunLogger(log_decisions=args.log_decisions)
        try:
            evaluate_model(model_file=model_file, run_logger=run_logger, **shared)
        finally:
            run_logger.close()
        print(f"\nEvaluation complete. Run logs written to: {run_logger.run_dir}")
    else:
        run_batch(pattern=args.pattern, log_decisions=args.log_decisions, **shared)
