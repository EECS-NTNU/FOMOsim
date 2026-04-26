"""
run_ablation_study.py  -  VFA Feature Ablation Study
 
Usage
-----
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
--------------------------------------------------------------
  Pillar 1 — Current System Imbalance (CIM, always active)
    CIM1  rebalancing_imbalance          total L1 deviation from target
    CIM2  squared_starvation_penalty     mean squared starvation ratio
    CIM3  squared_congestion_penalty     mean squared congestion ratio
    CIM4  exponential_starvation_penalty exponential starvation penalty, bounded [0,1]
    CIM5  exponential_congestion_penalty exponential congestion penalty, bounded [0,1]
    CIM6  starvation_severity_max        95th-percentile starvation ratio
    CIM7  congestion_severity_max        95th-percentile congestion ratio
    CIM8  starvation_variance            variance of starvation ratios
    CIM9  starvation_count               fraction of stations with starvation ratio >= 0.9
    CIM10 congestion_count               fraction of stations with congestion ratio >= 0.9

  Pillar 2 — Future System Imbalance (FIM, demand_horizon_enabled)
    FIM1  gross_starvation_risk          departure pressure vs current inventory (Poisson-adjusted)
    FIM2  gross_congestion_risk          arrival pressure vs free docks (Poisson-adjusted)
    FIM3  net_starvation_shortfall       net outflow pressure with demand uncertainty
    FIM4  net_congestion_shortfall       net inflow pressure with demand uncertainty

  Pillar 3 — Maintenance Pressure (MP, maintenance_enabled)
    MP1   trailer_cannibalization        broken bike fraction of van capacity
    MP2   global_onsite_backlog          onsite broken bikes normalised by fleet
    MP3   demand_weighted_depot_backlog  broken bikes at high-activity stations
    MP4   depot_pull                     urgency to return to depot
    MP5   maintenance_urgency            onsite backlog × starvation breadth (CIM9)

  Pillar 4 — Spatial & Logistic Constraints (SLC, logistics_enabled)
    SLC5  imbalance_hotspot_distance     distance to worst-imbalance stations, normalised
"""
 
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
 
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))
 
from policies.sjovik_sund.vfa.train_vfa import train, ALPHA_START
 
 
# -----------------------------------------------------------------------------
# Experiment definitions
#
# Each entry is a named subset of features to train the VFA with.
# Features must be valid names from vfa_features.get_feature_names().
# The training loop in LinearVFAPolicy enforces canonical ordering automatically,
# so the order listed here does not matter.
# -----------------------------------------------------------------------------
 
EXPERIMENTS = {
    # -- Squared_Temporal -----------------------------------------------------
    # Hypothesis: Squared penalties capture current-state imbalance depth and
    # symmetric temporal features cover both demand sides. Minimal and stable -
    # serves as the primary baseline against all other experiments.
    "Squared_Temporal": [
        "squared_starvation_penalty",     # CIM2: mean squared starvation ratio
        "squared_congestion_penalty",     # CIM3: mean squared congestion ratio
        "gross_starvation_risk",          # FIM1: gross departure pressure
        "gross_congestion_risk",          # FIM2: gross arrival pressure
    ],

    # -- Starvation_focused ----------------------------------------------------
    # Hypothesis: Starvation is the dominant failure mode in this instance, so the
    # VFA only needs to penalise starvation depth (CIM2/CIM6) and anticipate starvation
    # demand (FIM1). Congestion features dilute the gradient signal by forcing the
    # VFA to learn a trade-off that rarely matters in practice. Asymmetric by
    # design - if this outperforms Squared_Temporal, it confirms that congestion
    # features are noise rather than signal for this network.
    "Starvation_focused": [
        "squared_starvation_penalty",     # CIM2
        "squared_congestion_penalty",     # CIM3
        "starvation_severity_max",        # CIM6: Q95 starvation tail depth
        "gross_starvation_risk",          # FIM1: gross departure pressure
    ],


    # -- Short_term_only -------------------------------------------------------
    # Hypothesis: Temporal anticipation (FIM1/FIM2) is unnecessary - current-state
    # global mass (CIM1) combined with Q95 tail severity on both sides already
    # captures everything the VFA needs. If this performs comparably to
    # Squared_Temporal, temporal features provide no net value.
    "Short_term_only": [
        "rebalancing_imbalance",          # CIM1: total L1 imbalance across the network
        "starvation_severity_max",        # CIM6: Q95 starvation tail depth
        "congestion_severity_max",        # CIM7: Q95 congestion tail depth
    ],

    # -- Exponential_Temporal -------------------------------------------------
    # Hypothesis: Exponential penalties, which disproportionately punish deep
    # starvation/congestion, produce a better-shaped value surface than squared
    # penalties when combined with symmetric temporal anticipation. Directly
    # comparable to Squared_Temporal - same structure, different penalty form.
    "Exponential_Temporal": [
        "exponential_starvation_penalty", # CIM4: exp penalty, emphasises tail states
        "exponential_congestion_penalty", # CIM5: symmetric
        "gross_starvation_risk",          # FIM1: gross departure pressure
        "gross_congestion_risk",          # FIM2: gross arrival pressure
    ],

    # -- Count_Temporal -------------------------------------------------------
    # Hypothesis: Breadth (how many stations affected) carries equivalent signal
    # to depth (how badly affected). Direct comparison to Squared_Temporal.
    # If it matches, depth and breadth encode the same information for this network.
    "Count_Temporal": [
        "starvation_count",               # CIM9: fraction of stations with starvation ratio >= 0.9
        "congestion_count",               # CIM10: fraction of stations with congestion ratio >= 0.9
        "gross_starvation_risk",          # FIM1: gross departure pressure
        "gross_congestion_risk",          # FIM2: gross arrival pressure
    ],

    # -- Net_Demand_Temporal ---------------------------------------------------
    # Hypothesis: Net demand flow (departures minus arrivals) is more informative
    # than gross flow, because arriving bikes partially offset departures.
    # FIM3/FIM4 should outperform FIM1/FIM2 when arrivals meaningfully replenish
    # inventory within the horizon.
    "Net_Demand_Temporal": [
        "squared_starvation_penalty",     # CIM2: current depth anchor
        "squared_congestion_penalty",     # CIM3: symmetric
        "net_starvation_shortfall",       # FIM3: net outflow pressure
        "net_congestion_shortfall",       # FIM4: net inflow pressure
    ],

    # -- Severity_Temporal ----------------------------------------------------
    # Hypothesis: Q95 tail-severity features alone (without mean-penalty features)
    # are sufficient to distinguish good from bad states, and temporal features
    # give them forward-looking context. If this matches Squared_Temporal,
    # tail depth and mean depth carry equivalent information for the VFA.
    "Severity_Temporal": [
        "starvation_severity_max",        # CIM6: Q95 starvation tail depth
        "congestion_severity_max",        # CIM7: Q95 congestion tail depth
        "gross_starvation_risk",          # FIM1: gross departure pressure
        "gross_congestion_risk",          # FIM2: gross arrival pressure
    ],


        # -- Combined_Temporal ----------------------------------------------------
    # Hypothesis: Combining mean-penalty features (CIM2/CIM3) with tail-severity
    # features (CIM6/CIM7) gives the VFA both network-wide depth and worst-case
    # bottleneck signals simultaneously, outperforming either family alone.
    # If this does not beat Squared_Temporal, CIM6/CIM7 add no incremental value.
    "Combined_Temporal": [
        "squared_starvation_penalty",     # CIM2: mean squared starvation ratio
        "squared_congestion_penalty",     # CIM3: mean squared congestion ratio
        "starvation_severity_max",        # CIM6: Q95 starvation tail depth
        "congestion_severity_max",        # CIM7: Q95 congestion tail depth
        "gross_starvation_risk",          # FIM1: gross departure pressure
        "gross_congestion_risk",          # FIM2: gross arrival pressure
    ],
 
    # -- Disregarded experiments -----------------------------------------------
    # "Breadth_And_Mass": [
    #     # Dropped: hotspot_imbalance_mass converged to ~0 weight in all runs.
    #     "starvation_count",               # A15
    #     "congestion_count",               # A16
    #     "hotspot_imbalance_mass",         # A18
    # ],
    # "Micro_Severity_Spatial": [
    #     # Dropped: imbalance_hotspot_distance converged to ~0 weight in all runs.
    #     "starvation_severity_max",        # A9
    #     "congestion_severity_max",        # A10
    #     "imbalance_hotspot_distance",     # A14
    # ],
    # "FullVFA": [
    #     # Dropped: crashed at 64/200 episodes at alpha=0.1 due to too many correlated features.
    #     "exponential_starvation_penalty", # A5
    #     "exponential_congestion_penalty", # A6
    #     "squared_starvation_penalty",     # A3
    #     "squared_congestion_penalty",     # A4
    #     "starvation_severity_max",        # A9
    #     "congestion_severity_max",        # A10
    #     "imbalance_hotspot_distance",     # A14
    #     "rebalancing_imbalance",          # A1
    #     "starvation_count",               # A15
    #     "congestion_count",               # A16
    #     "hotspot_imbalance_mass",         # A18
    #     "gross_starvation_risk",      # D4
    #     "gross_congestion_risk",      # D5
    # ],
}
 
 
# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
 
def run_all_experiments(seeds: list[int], episodes: int = 200, run_only: Optional[list[str]] = None, alphas: Optional[list[float]] = None, output_dir: str = "results", weight_starvation: float = -1.0, weight_congestion: float = -1.0, gamma: float = 0.99):
    if run_only:
        unknown = set(run_only) - set(EXPERIMENTS)
        if unknown:
            raise ValueError(f"Unknown experiment(s): {unknown}. Valid: {list(EXPERIMENTS)}")
 
    if alphas is None:
        alphas = [ALPHA_START]
 
    experiments = {k: v for k, v in EXPERIMENTS.items() if run_only is None or k in run_only}
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    # 1. OUTERMOST LOOP: Seeds
    for run_id, seed_offset in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"STARTING SEED: {seed_offset}  (Run {run_id + 1}/{len(seeds)})")
        print(f"{'='*60}")
        
        # 2. MIDDLE LOOP: Alphas
        for alpha in alphas:
            print(f"\n  -> RUNNING WITH ALPHA: {alpha}")
            
            # 3. INNERMOST LOOP: Experiments
            for exp_name, features in experiments.items():
                print(f"\n{'='*40}")
                print(f"EXPERIMENT: {exp_name} | Alpha: {alpha} | Seed: {seed_offset}")
                for f in features:
                    print(f"  - {f}")
                print(f"{'='*40}")

                # Force 'models' to be the root, and add your custom folder name inside it
                base_dir = Path("models") / output_dir
                exp_dir = base_dir / f"{exp_name}_alpha_{alpha}_{timestamp}"
                
                # Safely create the whole chain (models -> custom_name -> exp_name)
                # exist_ok=True ensures subsequent seeds peacefully reuse this folder
                exp_dir.mkdir(parents=True, exist_ok=True)
 
                train(
                    num_episodes=episodes,
                    save_path=exp_dir / f"vfa_{exp_name}_seed{seed_offset}.pkl",
                    seed_offset=seed_offset,
                    active_features=features,
                    alpha_start=alpha,
                    gamma=gamma,
                    weight_starvation=weight_starvation,
                    weight_congestion=weight_congestion,
                )
 
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VFA feature ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available experiments: {', '.join(EXPERIMENTS)}",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default= [1000], metavar="SEED",
        help="Seed offsets to run (one independent training run per seed)",
    )
    parser.add_argument(
        "--episodes", type=int, default=350,
        help="Training episodes per run",
    )
    parser.add_argument(
        "--experiments", nargs="+", type=str, default=None, metavar="NAME",
        help="Subset of experiments to run (default: all)",
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.1], metavar="ALPHA",
        help="List of alpha (learning rate) values to test. e.g. --alphas 0.001 0.005 0.01",
    )

    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save experiment results"
    )
    parser.add_argument(
        "--weight_starvation", type=float, default=-1.0,
        help="Reward weight for starvation events (default: -1.0)",
    )
    parser.add_argument(
        "--weight_congestion", type=float, default=-1.0,
        help="Reward weight for congestion events (default: -1.0)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor (default: 0.99 per hour)",
    )

    args = parser.parse_args()
    run_all_experiments(
        seeds=args.seeds,
        episodes=args.episodes,
        run_only=args.experiments,
        alphas=args.alphas,
        output_dir=args.output_dir,
        weight_starvation=args.weight_starvation,
        weight_congestion=args.weight_congestion,
        gamma=args.gamma,
    )
 
    # -- Axis 2: Spatial Recoverability ---------------------------------------
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
 
    # -- Axis 3: Long-Term Recoverability -------------------------------------
    # Do features that encode how recoverable the state is (rather than just
    # how bad it is) improve tail value estimation?
 
    # "LongTerm": [
    #     "squared_starvation_penalty",     # A3: baseline - how bad is the state?
    #     "squared_congestion_penalty",     # A4
    # ],
 
    # -- Axis 4: Iterative Refinement -----------------------------------------
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
 
 
    # -- New Experiments (April) ----------------------------------------------
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
 
    # -- New Diverse Sets (April Refresh) ------------------------------------
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
 
        # -- Axis 1: Temporal Horizon Depth ---------------------------------------
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
 
 