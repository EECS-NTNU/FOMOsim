"""
run_ablation_study.py  —  VFA Feature Ablation Study

Usage
─────
    # Run all experiments, 3 seeds, 200 episodes each:
    python run_ablation_study.py

    # Run specific experiments only:
    python run_ablation_study.py --experiments V3_Rollout LongTerm_Only

    # Custom seeds and episode count:
    python run_ablation_study.py --seeds 1000 2000 3000 --episodes 300

Results are saved under:
    models/ablation_study/<experiment_name>/vfa_<name>_seed<N>.pkl
    models/ablation_study/<experiment_name>/vfa_<name>_seed<N>_weights_evolution.csv
    models/ablation_study/<experiment_name>/vfa_<name>_seed<N>_learning_curve.npy

Feature reference  (see vfa_features.py for full definitions)
──────────────────────────────────────────────────────────────
  Category A — Base Rebalancing (always available)
    A1  rebalancing_imbalance          total L1 deviation from target
    A2  anticipated_demand_shortfall   one-step starvation + congestion risk
    A3  squared_starvation_penalty     mean squared starvation depth
    A4  squared_congestion_penalty     mean squared congestion depth
    A5  proximity_to_demand_gravity    penalty for being far from demand
    A6  starvation_gravity             van load × proximity to starving stations
    A7  congestion_gravity             van free space × proximity to congested stations
    A8  imbalance_weighted_distance    distant imbalance = deferred cost
    A9  starvation_severity_max        worst single-station starvation ratio
    A10 congestion_severity_max        worst single-station congestion ratio
    A11 station_starvation_count       fraction of stations below target
    A12 work_ratio                     van trips needed to fix all imbalance
    A13 imbalance_concentration        tractability — is imbalance in one spot or everywhere?

  Category D — Temporal Demand (requires temporal_enabled=True in train_vfa.py)
    D1  time_of_day_fraction           where in the 24h cycle
    D4  multi_horizon_starvation_risk  integrated starvation over next H hours
"""

import os
import sys
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.train_vfa import train


# ─────────────────────────────────────────────────────────────────────────────
# Experiment definitions
#
# Each entry is a named subset of features to train the VFA with.
# Features must be valid names from vfa_features.get_feature_names().
# The training loop in LinearVFAPolicy enforces canonical ordering automatically,
# so the order listed here does not matter.
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {

    # ── Axis 1: Temporal Horizon Depth ───────────────────────────────────────
    # How far ahead does the VFA need to see to make good decisions?
    # These experiments isolate the temporal dimension from spatial features.

    # One-step demand anticipation only (no multi-hour look-ahead)
    "H1_OneStep": [
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "anticipated_demand_shortfall",   # A2: one-step activity
    ],

    # Multi-step demand integration using the target matrix
    "HN_MultiStep": [
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "multi_horizon_starvation_risk",  # D4: integrated over H hours
        "time_of_day_fraction",           # D1: anchors D4 in the daily cycle
    ],

    # ── Axis 2: Spatial Recoverability ───────────────────────────────────────
    # Does spatial structure (distance-weighted features, gravity) help
    # beyond pure magnitude features?

    "Spatial": [
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "imbalance_weighted_distance",    # A8: distant imbalance = deferred cost
        "starvation_severity_max",        # A9: worst-case bottleneck
        "starvation_gravity",             # A6: van load × proximity to starving stations
        "congestion_gravity",             # A7: van free space × proximity to congested stations
    ],

    # ── Axis 3: Long-Term Recoverability ─────────────────────────────────────
    # Do features that encode how recoverable the state is (rather than just
    # how bad it is) improve tail value estimation?

    "LongTerm": [
        "squared_starvation_penalty",     # A3: baseline — how bad is the state?
        "squared_congestion_penalty",     # A4
        "work_ratio",                     # A12: how many van trips to fix everything?
        "imbalance_concentration",        # A13: is the work concentrated or spread out?
    ],

    # ── Axis 4: Iterative Refinement ─────────────────────────────────────────
    # Progressive builds toward the best hypothesis, so results can be compared
    # incrementally rather than in a single jump.

    # V2 Refined Core: gravity replaces simple potentials; adds temporal features.
    # Hypothesis: fewer competing gradients → faster, more stable convergence.
    "V2_Core": [
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "starvation_severity_max",        # A9
        "starvation_gravity",             # A6
        "congestion_gravity",             # A7
        "imbalance_weighted_distance",    # A8
        "multi_horizon_starvation_risk",  # D4
        "time_of_day_fraction",           # D1
    ],

    # V2 Extended: V2_Core + symmetric congestion severity + starvation breadth.
    "V2_Extended": [
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "starvation_severity_max",        # A9
        "congestion_severity_max",        # A10
        "starvation_gravity",             # A6
        "congestion_gravity",             # A7
        "imbalance_weighted_distance",    # A8
        "station_starvation_count",       # A11
        "multi_horizon_starvation_risk",  # D4
        "time_of_day_fraction",           # D1
    ],

    # V3 Rollout: best-hypothesis set for rollout tail value.
    # V2_Core + long-term recoverability features (A12, A13) + global imbalance (A1).
    "V3_Rollout": [
        "rebalancing_imbalance",          # A1: total work remaining
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "starvation_severity_max",        # A9
        "congestion_severity_max",        # A10
        "starvation_gravity",             # A6
        "congestion_gravity",             # A7
        "imbalance_weighted_distance",    # A8
        "work_ratio",                     # A12: can the network be recovered?
        "imbalance_concentration",        # A13: tractability
        "multi_horizon_starvation_risk",  # D4
        "time_of_day_fraction",           # D1
    ],

    # ── Kitchen Sink ─────────────────────────────────────────────────────────
    # All non-maintenance features. Useful as an upper bound and to check for
    # harmful redundancy (if FullVFA is worse than V3_Rollout, some features hurt).
    "FullVFA": [
        "rebalancing_imbalance",          # A1
        "anticipated_demand_shortfall",   # A2
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "proximity_to_demand_gravity",    # A5
        "starvation_gravity",             # A6
        "congestion_gravity",             # A7
        "imbalance_weighted_distance",    # A8
        "starvation_severity_max",        # A9
        "congestion_severity_max",        # A10
        "station_starvation_count",       # A11
        "work_ratio",                     # A12
        "imbalance_concentration",        # A13
        "multi_horizon_starvation_risk",  # D4
        "time_of_day_fraction",           # D1
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(seeds: list[int], episodes: int = 200, run_only: list[str] = None):
    if run_only:
        unknown = set(run_only) - set(EXPERIMENTS)
        if unknown:
            raise ValueError(f"Unknown experiment(s): {unknown}. Valid: {list(EXPERIMENTS)}")

    experiments = {k: v for k, v in EXPERIMENTS.items() if run_only is None or k in run_only}

    base_dir = Path("models/ablation_study")
    base_dir.mkdir(parents=True, exist_ok=True)

    for exp_name, features in experiments.items():
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp_name}  ({len(features)} features)")
        for f in features:
            print(f"  - {f}")
        print(f"{'='*60}")

        exp_dir = base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)

        for run_id, seed_offset in enumerate(seeds):
            print(f"\n  Run {run_id + 1}/{len(seeds)}  (seed={seed_offset})")
            train(
                num_episodes=episodes,
                save_path=exp_dir / f"vfa_{exp_name}_seed{seed_offset}.pkl",
                seed_offset=seed_offset,
                active_features=features,
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VFA feature ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available experiments: {', '.join(EXPERIMENTS)}",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1000, 2000, 3000],
        help="Seed offsets to run (one independent training run per seed)",
    )
    parser.add_argument(
        "--episodes", type=int, default=200,
        help="Training episodes per run",
    )
    parser.add_argument(
        "--experiments", nargs="+", type=str, default=None, metavar="NAME",
        help="Subset of experiments to run (default: all)",
    )

    args = parser.parse_args()
    run_all_experiments(seeds=args.seeds, episodes=args.episodes, run_only=args.experiments)
