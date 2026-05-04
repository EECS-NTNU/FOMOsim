#!/usr/bin/env python3
"""
evaluate_hybrid_rollout.py  —  Evaluate trained VFA models inside the Hybrid Rollout Policy

Two modes of operation:

  1. BATCH MODE (iterates over all experiment x alpha combinations):
     For each combination, scans models/final_ablation_300ep/ for per-seed
     weights_evolution CSVs, averages the last --last_n episode rows across all
     available seeds, and evaluates the resulting averaged VFA inside a Hybrid
     Rollout Policy.

       # All experiments, both alphas, default 10-episode tail average:
       python evaluate_hybrid_rollout.py

       # Custom training seeds / alphas / tail window:
       python evaluate_hybrid_rollout.py --train_seeds 5000 6000 --train_alphas 0.1 0.05 --last_n 10

  2. SINGLE MODE:
     Point directly at one .pkl file. Features are auto-detected from the folder name,
     or overridden manually with --features.

       python evaluate_hybrid_rollout.py --model models/final_ablation_300ep/seed5000_alpha0.1_/Squared_Temporal_alpha_0.1_20260422_150154/vfa_Squared_Temporal_seed5000.pkl
       python evaluate_hybrid_rollout.py --model my_model.pkl --features squared_starvation_penalty projected_starvation_risk

Each evaluated model is compared against a DoNothing baseline.
Results are written to the standard simulation_results/csv output folder.
"""

import os
import sys
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
# run_logger import is deferred to avoid circular dependency at module load time
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.vfa.HybridRolloutPolicy import HybridRolloutPolicy
from policies.sjovik_sund.vfa.run_logger import RunLogger
from policies.do_nothing_policy import DoNothing
from policies.sjovik_sund.run_simulation_ingvild import SimulationConfig, test_policies

# Single source of truth for experiment definitions
from policies.sjovik_sund.ablation_study.run_ablation_study import EXPERIMENTS

FINAL_ABLATION_DIR = Path("models/final_ablation_500ep")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for CSV-based weight averaging
# ─────────────────────────────────────────────────────────────────────────────

def _find_weights_csv(
    base_dir: Path,
    exp_name: str,
    alpha: float,
    seed: int,
) -> Path | None:
    """Locate vfa_*_weights_evolution.csv for one (exp, alpha, seed) combination.

    Supports two directory layouts:
      - Nested:  base_dir/seed{seed}_alpha{alpha}_/{exp}_alpha_{alpha}_*/vfa_*_seed{seed}_*.csv
      - Flat:    base_dir/{exp}_alpha_{alpha}_*/vfa_*_seed{seed}_*.csv
    """
    alpha_str = str(alpha)
    csv_name = f"vfa_{exp_name}_seed{seed}_weights_evolution.csv"
    exp_prefix = f"{exp_name}_alpha_{alpha_str}_"

    # Nested layout (final_ablation_300ep style)
    seed_alpha_dir = base_dir / f"seed{seed}_alpha{alpha_str}_"
    if seed_alpha_dir.exists():
        for exp_dir in seed_alpha_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith(exp_prefix):
                csv = exp_dir / csv_name
                if csv.exists():
                    return csv

    # Flat layout (final_ablation_350ep style)
    for exp_dir in base_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith(exp_prefix):
            csv = exp_dir / csv_name
            if csv.exists():
                return csv

    return None


def _load_averaged_theta(csv_path: Path, last_n: int) -> tuple[np.ndarray, list[str]]:
    """Return (mean theta over last_n episodes, feature_names) from a weights CSV."""
    df = pd.read_csv(csv_path)
    weight_cols = [c for c in df.columns if c not in ("episode", "service_level")]
    theta = df[weight_cols].values[-last_n:].mean(axis=0)
    return theta, weight_cols


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation — one pre-built policy
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_file: Path,
    active_features: list[str],
    exp_name: str,
    seed: int,
    lookahead_minutes: float,
    num_scenarios: int,
    n_routing_candidates: int,
    n_time_steps: int,
    use_degradation: bool,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
    run_logger=None,
    debug_print: bool = False,
):
    print(f"\n{'='*60}")
    print(f"  {exp_name}  |  seed={seed}  |  {len(active_features)} features")
    print(f"  Model: {model_file.name}")
    print(f"{'='*60}")

    trained_vfa = LinearVFAPolicy.load(model_file, active_features=active_features)
    trained_vfa.learning_mode = False

    alpha_match = re.search(r'alpha_([\d.]+)', str(model_file))
    alpha_val = float(alpha_match.group(1)) if alpha_match else 0.0
    alpha_str = f"_A{alpha_match.group(1)}" if alpha_match else ""

    if run_logger is not None:
        run_logger.set_run_label(
            exp_name,
            alpha_val,
            policy_type="hybrid",
            duration_hours=duration_hours,
            num_vehicles=vehicles,
            instance_name=instance,
        )
        trained_vfa.logger = run_logger

    hybrid_policy = HybridRolloutPolicy(
        trained_vfa=trained_vfa,
        lookahead_minutes=lookahead_minutes,
        num_scenarios=num_scenarios,
        n_routing_candidates=n_routing_candidates,
        n_time_steps=n_time_steps,
        use_degradation=use_degradation,
        logger=run_logger,
        debug_print=debug_print,
    )

    policy_dict = {
        # f"{exp_name}_seed{seed}{alpha_str}_VFA_Only": trained_vfa,
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
        run_logger=run_logger,
        write_csv=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch mode — one hybrid rollout test per (experiment × alpha) combination,
#              weights averaged over last_n episodes across all available seeds
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    train_seeds: list[int],
    train_alphas: list[float],
    last_n: int,
    base_dir: Path,
    lookahead_minutes: float,
    num_scenarios: int,
    n_routing_candidates: int,
    n_time_steps: int,
    use_degradation: bool,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
    log_decisions: bool = False,
    run_only: list[str] | None = None,
    run_donoting: bool = False,
    run_vfa: bool = False,
    run_hybrid: bool = True,
    debug_print: bool = False,
):
    # One RunLogger for the entire batch — all experiments/alphas/seeds share it
    run_logger = RunLogger(log_decisions=log_decisions)

    if run_donoting:
        print(f"\n{'='*60}")
        print("  RUNNING BASELINE: DoNothing")
        print(f"{'='*60}")
        run_logger.set_run_label(
            "DoNothing_Baseline",
            0.0,
            policy_type="do-nothing",
            duration_hours=duration_hours,
            num_vehicles=vehicles,
            instance_name=instance,
            results_only=True,
        )
        test_policies(
            list_of_seeds=list(range(start_seed, start_seed + episodes)),
            policy_dict={"DoNothing_Baseline": DoNothing()},
            num_vehicles=vehicles,
            duration=duration_hours,
            use_multiprocessing=False,
            instance_name=instance,
            config=SimulationConfig(),
            run_logger=run_logger,
            write_csv=False,
        )

    if run_only:
        unknown = set(run_only) - set(EXPERIMENTS)
        if unknown:
            raise ValueError(f"Unknown experiment(s): {unknown}. Valid: {list(EXPERIMENTS)}")

    active_experiments = {k: v for k, v in EXPERIMENTS.items() if run_only is None or k in run_only}
    total = len(active_experiments) * len(train_alphas)
    run_idx = 0

    for alpha in train_alphas:
        alpha_str = str(alpha)
        for exp_name, _ in active_experiments.items():
            run_idx += 1
            print(f"\n{'='*60}")
            print(f"  [{run_idx}/{total}]  {exp_name}  |  alpha={alpha}")
            print(f"  Averaging last {last_n} episodes over seeds {train_seeds}")
            print(f"{'='*60}")

            seed_thetas: list[np.ndarray] = []
            feature_names: list[str] | None = None

            for seed in train_seeds:
                csv_path = _find_weights_csv(base_dir, exp_name, alpha, seed)
                if csv_path is None:
                    print(f"  [SKIP seed {seed}] CSV not found under {base_dir}")
                    continue
                theta, feat_names = _load_averaged_theta(csv_path, last_n)
                seed_thetas.append(theta)
                if feature_names is None:
                    feature_names = feat_names
                print(f"  [seed {seed}]  {csv_path.parent.name}")
                print(f"             θ = {np.round(theta, 4)}")

            if not seed_thetas:
                print(f"  [SKIP] No CSVs found for {exp_name} alpha={alpha} — skipping.")
                continue

            avg_theta = np.mean(np.stack(seed_thetas), axis=0)
            n_seeds = len(seed_thetas)
            print(f"  Averaged {n_seeds} seed(s):  θ = {np.round(avg_theta, 4)}")

            vfa = LinearVFAPolicy(
                active_features=feature_names,
                n_features=len(avg_theta),
                learning_mode=False,
            )
            vfa.theta = avg_theta.copy()
            vfa.weights = list(avg_theta)

            tag = f"{exp_name}_A{alpha_str}_last{last_n}ep_{n_seeds}seeds"
            eval_seeds = list(range(start_seed, start_seed + episodes))

            if run_vfa:
                run_logger.set_run_label(
                    exp_name,
                    alpha,
                    policy_type="vfa",
                    duration_hours=duration_hours,
                    num_vehicles=vehicles,
                    instance_name=instance,
                )
                vfa.logger = run_logger
                test_policies(
                    list_of_seeds=eval_seeds,
                    policy_dict={f"{tag}_VFA": vfa},
                    num_vehicles=vehicles,
                    duration=duration_hours,
                    use_multiprocessing=False,
                    instance_name=instance,
                    config=SimulationConfig(),
                    run_logger=run_logger,
                    write_csv=False,
                )

            if run_hybrid:
                run_logger.set_run_label(
                    exp_name,
                    alpha,
                    policy_type="hybrid",
                    duration_hours=duration_hours,
                    num_vehicles=vehicles,
                    instance_name=instance,
                )
                vfa.logger = run_logger
                hybrid = HybridRolloutPolicy(
                    trained_vfa=vfa,
                    lookahead_minutes=lookahead_minutes,
                    num_scenarios=num_scenarios,
                    n_routing_candidates=n_routing_candidates,
                    n_time_steps=n_time_steps,
                    use_degradation=use_degradation,
                    logger=run_logger,
                    debug_print=debug_print,
                )

                test_policies(
                    list_of_seeds=eval_seeds,
                    policy_dict={f"{tag}_Hybrid_H{int(lookahead_minutes)}_S{num_scenarios}": hybrid},
                    num_vehicles=vehicles,
                    duration=duration_hours,
                    use_multiprocessing=False,
                    instance_name=instance,
                    config=SimulationConfig(),
                    run_logger=run_logger,
                    write_csv=False,
                )

    run_logger.close()
    print(f"\n{'='*60}")
    print(f"Batch complete ({run_idx} runs).")
    print(f"  Run logs: {run_logger.run_dir}")
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
    n_routing_candidates: int,
    n_time_steps: int,
    use_degradation: bool,
    episodes: int,
    start_seed: int,
    duration_hours: int,
    instance: str,
    vehicles: int,
    log_decisions: bool = False,
    run_donoting: bool = False,
    run_vfa: bool = False,
    run_hybrid: bool = True,
    debug_print: bool = False,
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

    alpha_match = re.search(r'alpha_([\d.]+)', model_path)
    alpha_val = float(alpha_match.group(1)) if alpha_match else 0.0

    run_logger = RunLogger(log_decisions=log_decisions)
    eval_seeds = list(range(start_seed, start_seed + episodes))

    if run_donoting:
        print(f"\n{'='*60}")
        print("  RUNNING BASELINE: DoNothing")
        print(f"{'='*60}")
        run_logger.set_run_label(
            "DoNothing_Baseline",
            0.0,
            policy_type="do-nothing",
            duration_hours=duration_hours,
            num_vehicles=vehicles,
            instance_name=instance,
            results_only=True,
        )
        test_policies(
            list_of_seeds=eval_seeds,
            policy_dict={"DoNothing_Baseline": DoNothing()},
            num_vehicles=vehicles,
            duration=duration_hours,
            use_multiprocessing=False,
            instance_name=instance,
            config=SimulationConfig(),
            run_logger=run_logger,
            write_csv=False,
        )

    if run_vfa or run_hybrid:
        trained_vfa = LinearVFAPolicy.load(model_file, active_features=active_features)
        trained_vfa.learning_mode = False

        if run_vfa:
            print(f"\n{'='*60}")
            print(f"  RUNNING VFA-only: {exp_name}")
            print(f"{'='*60}")
            run_logger.set_run_label(
                exp_name,
                alpha_val,
                policy_type="vfa",
                duration_hours=duration_hours,
                num_vehicles=vehicles,
                instance_name=instance,
            )
            trained_vfa.logger = run_logger
            test_policies(
                list_of_seeds=eval_seeds,
                policy_dict={f"{exp_name}_VFA": trained_vfa},
                num_vehicles=vehicles,
                duration=duration_hours,
                use_multiprocessing=False,
                instance_name=instance,
                config=SimulationConfig(),
                run_logger=run_logger,
                write_csv=False,
            )

        if run_hybrid:
            evaluate_model(
                model_file=model_file,
                active_features=active_features,
                exp_name=exp_name,
                seed=0,
                lookahead_minutes=lookahead_minutes,
                num_scenarios=num_scenarios,
                n_routing_candidates=n_routing_candidates,
                n_time_steps=n_time_steps,
                use_degradation=use_degradation,
                episodes=episodes,
                start_seed=start_seed,
                duration_hours=duration_hours,
                instance=instance,
                vehicles=vehicles,
                run_logger=run_logger,
                debug_print=debug_print,
            )

    run_logger.close()
    print(f"\nRun logs written to: {run_logger.run_dir}")


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
    mode = parser.add_argument_group("Mode (omit --model for batch mode)")
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
        "--train_seeds", nargs="+", type=int, default=[1000], metavar="SEED",
        help="Training seeds whose weight CSVs to average (default: 5000 6000)",
    )
    batch.add_argument(
        "--train_alphas", nargs="+", type=float, default=[0.1], metavar="ALPHA",
        help="Training alphas to evaluate (default: 0.1)",
    )
    batch.add_argument(
        "--last_n", type=int, default=5, metavar="N",
        help="Average weights over last N episodes per seed (default: 10)",
    )
    batch.add_argument(
        "--base_dir", type=str, default=str(FINAL_ABLATION_DIR),
        help=f"Root directory containing trained models (default: {FINAL_ABLATION_DIR})",
    )

    batch.add_argument(
        "--run_only", nargs="+", type=str, default=None, metavar="NAME",
        help="Subset of experiments to run, e.g. --run_only Squared_Temporal Short_term_only",
    )
    batch.add_argument(
        "--log_decisions", action="store_true", default=False,
        help="Write decisions.csv (one row per real vehicle decision). Can be large for long runs.",
    )
    batch.add_argument(
        "--debug", action="store_true", default=False,
        help="Print per-decision candidate tables: pre-pruning VFA ranking and post-screening survivors.",
    )
    batch.add_argument(
        "--run_donoting", action="store_true", default=False,
        help="Also run the DoNothing baseline (results.csv only, no hourly/daily metrics).",
    )
    batch.add_argument(
        "--run_vfa", action="store_true", default=False,
        help="Also run the VFA-only policy for each experiment before the hybrid rollout.",
    )
    batch.add_argument(
        "--no_hybrid", action="store_true", default=False,
        help="Skip the hybrid rollout (useful for running baselines only).",
    )

    # Rollout tuning
    rollout = parser.add_argument_group("Rollout parameters")
    rollout.add_argument("--lookahead", type=float, default=60.0,
                         help="Rollout horizon in simulation minutes (default: 60)")
    rollout.add_argument("--scenarios", type=int, default=10,
                         help="Monte Carlo scenarios per action (default: 10)")
    rollout.add_argument("--n_routing", type=int, default=10,
                         help="Routing targets per operational profile for initial candidate generation (default: 10)")
    rollout.add_argument("--n_time_steps", type=int, default=4,
                         help="Sub-intervals per rollout horizon for demand sampling (default: 4 × 15 min for H=60)")
    rollout.add_argument("--use_degradation", action="store_true", default=False,
                         help="Enable Weibull component-failure sampling during rollout (default: off)")

    # Simulation settings
    sim = parser.add_argument_group("Simulation settings")
    sim.add_argument("--episodes", type=int, default=5,
                     help="Evaluation episodes per model (default: 1)")
    sim.add_argument("--seed", type=int, default=42,
                     help="Starting evaluation seed (default: 9000, kept separate from training seeds)")
    sim.add_argument("--duration", type=int, default=24 * 7,
                     help="Simulation duration in hours (default: 336)")
    sim.add_argument("--instance", type=str, default="TD_W34_old",
                     help="Simulator instance name")
    sim.add_argument("--vehicles", type=int, default=1,
                     help="Number of service vehicles")

    args = parser.parse_args()

    shared_sim = dict(
        lookahead_minutes=args.lookahead,
        num_scenarios=args.scenarios,
        n_rollout_candidates=args.n_candidates,
        n_routing_candidates=args.n_routing,
        screening_mode=args.screening,
        n_screening_scenarios=args.n_screening,
        n_survivors=args.n_survivors,
        n_rollout_candidates=args.n_candidates,
        n_routing_candidates=args.n_routing,
        n_time_steps=args.n_time_steps,
        use_degradation=args.use_degradation,
        episodes=args.episodes,
        start_seed=args.seed,
        duration_hours=args.duration,
        instance=args.instance,
        vehicles=args.vehicles,
    )

    if args.model:
        run_single(model_path=args.model, features_override=args.features,
                   log_decisions=args.log_decisions, debug_print=args.debug,
                   run_donoting=args.run_donoting, run_vfa=args.run_vfa,
                   run_hybrid=not args.no_hybrid, **shared_sim)
        run_single(model_path=args.model, features_override=args.features,
                   log_decisions=args.log_decisions, debug_print=args.debug,
                   run_donoting=args.run_donoting, run_vfa=args.run_vfa,
                   run_hybrid=not args.no_hybrid, **shared_sim)
    else:
        run_batch(
            train_seeds=args.train_seeds,
            train_alphas=args.train_alphas,
            last_n=args.last_n,
            base_dir=Path(args.base_dir),
            log_decisions=args.log_decisions,
            run_only=args.run_only,
            run_donoting=args.run_donoting,
            run_vfa=args.run_vfa,
            run_hybrid=not args.no_hybrid,
            debug_print=args.debug,
            **shared_sim,
        )
