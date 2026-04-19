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
    A3  squared_starvation_penalty     mean squared starvation depth
    A4  squared_congestion_penalty     mean squared congestion depth
        A9  starvation_severity_max        95th-percentile starvation ratio
        A10 congestion_severity_max        95th-percentile congestion ratio
    A11 unmet_starvation_deficit       unmet starvation normalized by half capacity
    A12 starvation_variance            variance of starvation ratios

  Category D — Temporal Demand (requires temporal_enabled=True in train_vfa.py)
    D1  time_of_day_fraction           where in the 24h cycle
    D4  multi_horizon_starvation_risk  integrated starvation over next H hours
    D5  multi_horizon_congestion_risk  integrated congestion over next H hours
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.train_vfa import train, ALPHA_START


# ─────────────────────────────────────────────────────────────────────────────
# Experiment definitions
#
# Each entry is a named subset of features to train the VFA with.
# Features must be valid names from vfa_features.get_feature_names().
# The training loop in LinearVFAPolicy enforces canonical ordering automatically,
# so the order listed here does not matter.
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {

    # ── Kitchen Sink ─────────────────────────────────────────────────────────
    # Superset of all independent features tested across the 6 core experiments below.
    "FullVFA": [
        "exponential_starvation_penalty", # A5
        "exponential_congestion_penalty", # A6
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "starvation_severity_max",        # A9
        "congestion_severity_max",        # A10
        "unmet_starvation_deficit",       # A11
        "imbalance_hotspot_distance",     # A14
        "rebalancing_imbalance",          # A1
        "starvation_count",               # A15
        "congestion_count",               # A16
        "hotspot_imbalance_mass",         # A18
        "projected_starvation_risk",      # D4
        "projected_congestion_risk",      # D5
    ],

    # 1. Pure Exponential Baseline
    # Tests if heavily penalizing extreme cases (exponential) is sufficient on its own,
    # without diluting the signal with global averages.
    "Base_Exponential": [
        "exponential_starvation_penalty", # A5
        "exponential_congestion_penalty", # A6
    ],

    # 2. Micro Severity & Spatial Routing
    # Ignores average system health entirely. Tells the VFA to focus only on the 
    # worst individual stations (Q95) and how far away the biggest hotspot is.
    "Micro_Severity_Spatial": [
        "starvation_severity_max",        # A9
        "congestion_severity_max",        # A10
        "imbalance_hotspot_distance",     # A14
    ],

    # 3. Macro vs Anticipatory (Future Outlook)
    # Pits the current total global imbalance against our new OD-matrix projections.
    # 'rebalancing_imbalance' is used as a single proxy for current state since it 
    # perfectly correlates with most other current-state metrics.
    "Macro_vs_Future": [
        "rebalancing_imbalance",          # A1
        "projected_starvation_risk",      # D4
        "projected_congestion_risk",      # D5
    ],

    # 4. Breadth and Mass Concentration
    # Instead of looking at "how deep" the starvation is, this focuses on "how wide" 
    # the problem has spread, and how concentrated it is in hotspots.
    "Breadth_And_Mass": [
        "starvation_count",               # A15
        "congestion_count",               # A16
        "hotspot_imbalance_mass",         # A18
    ],

    # 5. Optimal Orthogonal Mix (Best Hypothesis)
    # The ultimate minimally-correlated combination. It takes exactly one feature 
    # from every independent mathematical cluster discovered in the correlation study.
    "Optimal_Orthogonal_Mix": [
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "imbalance_hotspot_distance",     # A14
        "projected_starvation_risk",      # D4
        "projected_congestion_risk",      # D5
    ],

    # 6. Structural Capacity Limits
    # Tests if the VFA can prioritize based purely on physical van constraints 
    # combined with future demand risks.
    "Structural_Constraints": [
        "unmet_starvation_deficit",       # A11
        "starvation_severity_max",        # A9
        "projected_starvation_risk",      # D4
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(seeds: list[int], episodes: int = 200, run_only: list[str] = None, alphas: list[float] = None):
    if run_only:
        unknown = set(run_only) - set(EXPERIMENTS)
        if unknown:
            raise ValueError(f"Unknown experiment(s): {unknown}. Valid: {list(EXPERIMENTS)}")

    if alphas is None:
        alphas = [ALPHA_START]

    experiments = {k: v for k, v in EXPERIMENTS.items() if run_only is None or k in run_only}

    base_dir = Path("models/ablation_study_solstorm")
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"RUNNING WITH ALPHA: {alpha}")
        print(f"{'='*60}")
        for exp_name, features in experiments.items():
            print(f"\n{'='*40}")
            print(f"EXPERIMENT: {exp_name}  ({len(features)} features)")
            for f in features:
                print(f"  - {f}")
            print(f"{'='*40}")

            exp_dir = base_dir / f"{exp_name}_alpha_{alpha}_{timestamp}"
            exp_dir.mkdir(exist_ok=True)

            for run_id, seed_offset in enumerate(seeds):
                print(f"\n  Run {run_id + 1}/{len(seeds)}  (seed={seed_offset})")
                train(
                    num_episodes=episodes,
                    save_path=exp_dir / f"vfa_{exp_name}_seed{seed_offset}.pkl",
                    seed_offset=seed_offset,
                    active_features=features,
                    alpha_start=alpha,
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
        "--seeds", nargs="+", type=int, default=[1000],
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
    parser.add_argument(
        "--alphas", nargs="+", type=float, default= [0.005,0.01],
        help="List of alpha (learning rate) values to test. e.g. --alphas 0.001 0.005 0.01",
    )

    args = parser.parse_args()
    run_all_experiments(
        seeds=args.seeds, 
        episodes=args.episodes, 
        run_only=args.experiments, 
        alphas=args.alphas
    )


    # ── Axis 2: Spatial Recoverability ───────────────────────────────────────
    # Does spatial structure (distance-weighted features, gravity) help
    # beyond pure magnitude features?
    # "Squared": [
    #     "squared_starvation_penalty", # A3
    #     "squared_congestion_penalty", # A4
    # ],

    # "Exponential": [
    #     "exponential_starvation_penalty", # A5
    #     "exponential_congestion_penalty", # A6
    # ],

    # ── Axis 3: Long-Term Recoverability ─────────────────────────────────────
    # Do features that encode how recoverable the state is (rather than just
    # how bad it is) improve tail value estimation?

    # "LongTerm": [
    #     "squared_starvation_penalty",     # A3: baseline — how bad is the state?
    #     "squared_congestion_penalty",     # A4
    # ],

    # ── Axis 4: Iterative Refinement ─────────────────────────────────────────
    # Progressive builds toward the best hypothesis, so results can be compared
    # incrementally rather than in a single jump.

    # V2 Extended: V2_Core + symmetric congestion severity + starvation breadth.
    # "V2_Extended": [
    #     "squared_starvation_penalty",     # A3
    #     "squared_congestion_penalty",     # A4
    #     "starvation_severity_max",        # A9
    #     "congestion_severity_max",        # A10
    #     "multi_horizon_starvation_risk",  # D4
    #     "time_of_day_fraction",           # D1
    # ],

    # V3 Rollout: best-hypothesis set for rollout tail value.
    # V2_Core + long-term recoverability features (A12, A14) + global imbalance (A1).
    # "V3_Rollout": [
    #     "rebalancing_imbalance",          # A1: total work remaining
    #     "squared_starvation_penalty",     # A3
    #     "squared_congestion_penalty",     # A4
    #     "starvation_severity_max",        # A9
    #     "congestion_severity_max",        # A10
    #     "multi_horizon_starvation_risk",  # D4
    #     "time_of_day_fraction",           # D1
    # ],


    # ── New Experiments (April) ──────────────────────────────────────────────
    # "LongTerm_Spatial": [
    #     "squared_starvation_penalty",     # A3
    #     "squared_congestion_penalty",     # A4
    #     "starvation_severity_max",        # A9: worst-case bottleneck
    # ],
    
    # Pure Macro: Strip away local severity depth and variance. Focus entirely 
    # on network mass and deferred future network mass.
    # "Pure_Macro": [
    #     "rebalancing_imbalance",          # A1: global mass
    #     "multi_horizon_starvation_risk",  # D4: future global mass
    #     "time_of_day_fraction",           # D1: temporal anchor
    # ],

    # Symmetric Temporal: Look at both horizon risks to avoid dropping off 
    # bikes at stations that have massive impending inflow.
    # "Symmetric_Temporal": [
    #     "squared_starvation_penalty",     # A3
    #     "squared_congestion_penalty",     # A4
    #     "multi_horizon_starvation_risk",  # D4
    #     "multi_horizon_congestion_risk",  # D5
    #     "time_of_day_fraction",           # D1
    # ],

    # Van State Terminal: Focus on whether the VFA accurately leaves the 
    # van in a state equipped to handle the residual starvation.
    # "Van_State_Terminal": [
    #     "squared_starvation_penalty",     # A3
    #     "squared_congestion_penalty",     # A4
    #     "unmet_starvation_deficit",       # A11: Is the van equipped to fix the remaining starvation?
    # ],

    # ── New Diverse Sets (April Refresh) ────────────────────────────────────
    # Built to reduce overlap and test compact hypotheses.

    # "Spatial_Recovery_Pressure": [
    #     "squared_starvation_penalty",     # A3
    #     "squared_congestion_penalty",     # A4
    #     "unmet_starvation_deficit",       # A11
    #     "imbalance_hotspot_distance",     # A14
    #     "starvation_count",               # A15
    #     "congestion_count",               # A16
    #     "future_net_pressure",            # D7
    #     "time_of_day_fraction",           # D1
    # ],

    # "V2_Core_HotspotBalance": [
    #     "squared_starvation_penalty",     # A3
    #     "squared_congestion_penalty",     # A4
    #     "starvation_severity_max",        # A9
    #     "imbalance_asymmetry",            # A17
    #     "hotspot_imbalance_mass",         # A18
    #     "multi_horizon_starvation_risk",  # D4
    #     "time_of_day_fraction",           # D1
    # ],

    # "Expo_Macro_Reachability": [
    #     "exponential_starvation_penalty", # A5
    #     "exponential_congestion_penalty", # A6
    #     "congestion_severity_max",        # A10
    #     "rebalancing_imbalance",          # A1
    #     "hotspot_imbalance_mass",         # A18
    #     "multi_horizon_congestion_risk",  # D5
    # ],

    # "Rollout_Balanced_MinRedundancy": [
    #     "squared_starvation_penalty",     # A3
    #     "congestion_severity_max",        # A10
    #     "congestion_count",               # A16
    #     "imbalance_asymmetry",            # A17
    #     "imbalance_hotspot_distance",     # A14
    #     "multi_horizon_starvation_risk",  # D4
    #     "temporal_demand_gradient",       # D6
    # ],

        # ── Axis 1: Temporal Horizon Depth ───────────────────────────────────────
    # How far ahead does the VFA need to see to make good decisions?
    # These experiments isolate the temporal dimension from spatial features.

    # One-step demand anticipation only (no multi-hour look-ahead)
    # "H1_OneStep": [
    #     "squared_starvation_penalty",     # A3
    #     "squared_congestion_penalty",     # A4
    # ],

    # Multi-step demand integration using the target matrix
    # "HN_MultiStep": [
    #     "squared_starvation_penalty",     # A3
    #     "squared_congestion_penalty",     # A4
    #     "multi_horizon_starvation_risk",  # D4: integrated over H hours
    #     "time_of_day_fraction",           # D1: anchors D4 in the daily cycle
    # ],
