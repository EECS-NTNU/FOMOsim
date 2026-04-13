import os
import sys
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.vfa.train_vfa import train

# Define your experimental subsets here!
EXPERIMENTS = {
    # ── AXIS 1: Temporal Horizon Depth ────────────────────────────────────────
    # Controls how far ahead the VFA "sees". Baselines for comparison.

    # Horizon-0: Pure reactive snapshot (no temporal info)
    "H0": [
        "rebalancing_imbalance",
    ],

    # Horizon-1: One-step demand anticipation
    "H1": [
        "squared_starvation_penalty",    # uses time-indexed target
        "squared_congestion_penalty",
        "anticipated_demand_shortfall",  # one-step activity
    ],

    # Horizon-N: Multi-step demand integration
    "HN": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "multi_horizon_starvation_risk", # integrate over H hours
        "time_of_day_fraction",
    ],

    # ── AXIS 2: Spatial Recoverability ────────────────────────────────────────
    # Tests whether spatial structure (distance-weighted imbalance, gravity)
    # adds over pure magnitude features.
    "SR": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "imbalance_weighted_distance",   # φ_r1: distant imbalance = deferred cost
        "starvation_severity_max",       # φ_r2: worst-case bottleneck
        "starvation_gravity",            # φ_6a: van load × proximity to starving stations
        "congestion_gravity",            # φ_6b: van free space × proximity to congested stations
    ],

    # ── AXIS 3: Full Candidate ─────────────────────────────────────────────────
    # Kitchen-sink test of all non-temporal features.
    "FullVFA": [
        "rebalancing_imbalance",
        "anticipated_demand_shortfall",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "proximity_to_demand_gravity",
        "starvation_gravity",
        "congestion_gravity",
        "imbalance_weighted_distance",
        "starvation_severity_max",
        "congestion_severity_max",
        "station_starvation_count",
        "work_ratio",
        "imbalance_concentration",
        "multi_horizon_starvation_risk",
        "time_of_day_fraction",
    ],

    # ── AXIS 4: V2 Refined Core ────────────────────────────────────────────────
    # Drops vehicle_functional_load and rebalancing_imbalance.
    # Gravity features replace the simpler delivery/pickup potentials.
    # Hypothesis: fewer competing gradients → faster, more stable convergence.
    "V2_RC": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "multi_horizon_starvation_risk",
        "time_of_day_fraction",
        "starvation_severity_max",
        "starvation_gravity",
        "congestion_gravity",
        "imbalance_weighted_distance",
    ],

    # ── AXIS 5: V2 Extended ────────────────────────────────────────────────────
    # V2_RC + symmetric congestion severity + starvation breadth.
    "V2_Extended": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "multi_horizon_starvation_risk",
        "time_of_day_fraction",
        "starvation_severity_max",
        "starvation_gravity",
        "congestion_gravity",
        "imbalance_weighted_distance",
        "congestion_severity_max",
        "station_starvation_count",
    ],

    # ── AXIS 6: Long-Term Features ─────────────────────────────────────────────
    # Tests the two new long-term features in isolation.
    # Hypothesis: work_ratio and imbalance_concentration improve tail value
    # by encoding recoverability rather than just the size of the problem.
    "LongTerm_Only": [
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "work_ratio",                    # trips needed to fix all imbalance
        "imbalance_concentration",       # fraction held by worst station (tractability)
    ],

    # ── AXIS 7: V3 Rollout Core ────────────────────────────────────────────────
    # Best-hypothesis set for rollout tail value.
    # Combines the proven V2 core with the new long-term features.
    "V3_Rollout": [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "starvation_severity_max",
        "congestion_severity_max",
        "starvation_gravity",
        "congestion_gravity",
        "imbalance_weighted_distance",
        "work_ratio",
        "imbalance_concentration",
        "multi_horizon_starvation_risk",
        "time_of_day_fraction",
    ],
}

def run_all_experiments(seeds: list[int], episodes: int = 200, run_only: list[str] = None, alpha_start: float = 0.5):
    base_dir = Path("models/ablation_study")
    base_dir.mkdir(parents=True, exist_ok=True)

    experiments = {k: v for k, v in EXPERIMENTS.items() if run_only is None or k in run_only}

    if run_only:
        unknown = set(run_only) - set(EXPERIMENTS)
        if unknown:
            raise ValueError(f"Unknown experiment(s): {unknown}. Valid: {list(EXPERIMENTS)}")

    for exp_name, features in experiments.items():
        print(f"\n{'='*60}")
        print(f"STARTING EXPERIMENT: {exp_name}")
        print(f"Features: {features}")
        print(f"Alpha: {alpha_start}")
        print(f"{'='*60}")

        exp_dir = base_dir / f"{exp_name}_alpha_{alpha_start}"
        exp_dir.mkdir(exist_ok=True)

        for run_id, seed_offset in enumerate(seeds):
            print(f"  --> Run {run_id + 1}/{len(seeds)} (Seed Offset: {seed_offset})")

            save_path = exp_dir / f"vfa_{exp_name}_seed{seed_offset}.pkl"

            train(
                num_episodes=episodes,
                save_path=save_path,
                seed_offset=seed_offset,
                active_features=features,
                alpha_start=alpha_start,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VFA Feature Ablation Study")

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1000, 2000, 3000],
        help="List of seed offsets to run for robust averaging (e.g., --seeds 1000 2000 3000)"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of training episodes per run"
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        default=None,
        metavar="NAME",
        help=f"Which experiments to run. Use 'all' or omit to run all. Choices: {list(EXPERIMENTS)}"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Initial learning rate (alpha) for TD updates (default: 0.5)"
    )

    args = parser.parse_args()

    run_only = None if (args.experiments is None or args.experiments == ["all"]) else args.experiments
    run_all_experiments(seeds=args.seeds, episodes=args.episodes, run_only=run_only, alpha_start=args.alpha)
