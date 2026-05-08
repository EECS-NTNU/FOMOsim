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
from typing import Optional, List
 
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))
 
from policies.sjovik_sund.vfa.train_vfa import train, ALPHA_START, EPSILON_START, EPSILON_END
 
 
# -----------------------------------------------------------------------------
# Experiment definitions
#
# Each entry is a named subset of features to train the VFA with.
# Features must be valid names from vfa_features.get_feature_names().
# The training loop in LinearVFAPolicy enforces canonical ordering automatically,
# so the order listed here does not matter.
# -----------------------------------------------------------------------------
 
EXPERIMENTS = {
    # -- KS_Filtered -----------------------------------------------------------
    # Derived from kitchen-sink analysis (38-feature run, 300 ep).
    # Drops features that were: near-zero weight (maintenance_urgency),
    # r>0.98 duplicates (exponential_*/net_*_shortfall, starvation/congestion_count,
    # destination_starv/cong_ratio, destination_roi_starvation, recoverable_starvation),
    # or purely noisy (trailer_cannibalization, destination_cong_ratio).
    # Keeps one representative per correlated cluster and the best-signal
    # maintenance feature from each sub-group.
    "KS_Filtered": [
        "rebalancing_imbalance",          # CIM1: total L1 imbalance
        "squared_starvation_penalty",     # CIM2: mean squared starvation ratio
        "squared_congestion_penalty",     # CIM3: mean squared congestion ratio
        "starvation_severity_max",        # CIM6: Q95 starvation tail depth
        "congestion_severity_max",        # CIM7: Q95 congestion tail depth
        "gross_starvation_risk",          # FIM1: gross departure pressure
        "gross_congestion_risk",          # FIM2: gross arrival pressure
        "demand_weighted_onsite_backlog", # MP: onsite backlog weighted by demand
        "demand_weighted_depot_backlog",  # MP: depot backlog weighted by demand
        "fleet_broken_fraction",          # MP: overall fleet health
        "depot_idle_fraction",            # MP: depot throughput signal
    ],

    #BASELINES
    "Imbalance": ["rebalancing_imbalance"], # CIM1: total L1 imbalance across the network
    "Imbalance_squared" : ["rebalancing_imbalance", "squared_starvation_penalty", "squared_congestion_penalty"], # CIM1 + CIM2/CIM3: global mass + mean squared depth
    "Squared_temporal": [
        "squared_starvation_penalty",     # CIM2: mean squared starvation ratio
        "squared_congestion_penalty",     # CIM3: mean squared congestion ratio
        "gross_starvation_risk",          # FIM1: gross departure pressure
        "gross_congestion_risk",          # FIM2: gross arrival pressure
    ],

    "Squared_only": [
        "squared_starvation_penalty",     # CIM2: mean squared starvation ratio
        "squared_congestion_penalty",     # CIM3: mean squared congestion ratio
    ],

    "Imbalance_temporal": [
        "rebalancing_imbalance",          # CIM1: total L1 imbalance across the network
        "gross_starvation_risk",          # FIM1: gross departure pressure
        "gross_congestion_risk",          # FIM2: gross arrival pressure
    ],

    "Squared_only": [
        "squared_starvation_penalty",     # CIM2: mean squared starvation ratio
        "squared_congestion_penalty",     # CIM3: mean squared congestion ratio
    ],
    
    "Imbalance_squared_temporal" : ["rebalancing_imbalance", "squared_starvation_penalty", "squared_congestion_penalty", "gross_starvation_risk", "gross_congestion_risk"], # CIM1 + FIM1/FIM2: global mass + gross departure/arrival pressure
    "Imbalance_severity_temporal": ["rebalancing_imbalance", "starvation_severity_max", "congestion_severity_max", "gross_starvation_risk", "gross_congestion_risk"], # CIM1 + CIM6/CIM7 + FIM1/FIM2: global mass + Q95 tail severity + gross departure/arrival pressure    
    "Imbalance_severity_squared_temporal": ["rebalancing_imbalance", "squared_starvation_penalty", "squared_congestion_penalty", "starvation_severity_max", "congestion_severity_max", "gross_starvation_risk", "gross_congestion_risk"], # CIM1 + CIM2/CIM3 + CIM6/CIM7 + FIM1/FIM2: global mass + mean squared depth + Q95 tail severity + gross departure/arrival pressure

    # -- Screening-guided rebalancing experiments -----------------------------
    # R4: Tests whether net temporal shortfall is more useful than gross
    # departure/arrival pressure when current imbalance depth is already known.
    "R4_NetTemporalAlternative": [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "net_starvation_shortfall",
        "net_congestion_shortfall",
    ],

    # R6: Tests whether the breadth of affected stations adds signal beyond the
    # usual global imbalance and squared depth features.
    "R6_BreadthDepth": [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "starvation_count",
        "congestion_count",
    ],

    # R7: Tests whether destination-local information improves action
    # discrimination beyond the compact global temporal core.
    "R7_DestinationAware": [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "gross_starvation_risk",
        "gross_congestion_risk",
        "destination_starv_ratio",
        "destination_cong_ratio",
        "destination_travel_penalty",
        "cur_station_func_deficit",
    ],

    "KS_Filtered_V2": [
        "rebalancing_imbalance",          # CIM1: total L1 imbalance
        "squared_starvation_penalty",     # CIM2: mean squared starvation ratio
        "squared_congestion_penalty",     # CIM3: mean squared congestion ratio
        "gross_starvation_risk",          # FIM1: gross departure pressure
        "gross_congestion_risk",          # FIM2: gross arrival pressure
        "demand_weighted_onsite_backlog", # MP: onsite backlog weighted by demand
        "demand_weighted_depot_backlog",  # MP: depot backlog weighted by demand
        "fleet_broken_fraction",          # MP: overall fleet health
        "depot_idle_fraction",            # MP: depot throughput signal
    ],

    #"Severity_only": ["starvation_severity_max", "congestion_severity_max"], # CIM6/CIM7: Q95 tail severity without global mass signal
    #"Imbalance_severity": ["rebalancing_imbalance", "starvation_severity_max", "congestion_severity_max"], # CIM1 + CIM6/CIM7: global mass + Q95 tail severity      
    
    
    # ① vs ③: does depth add value on top of CIM1?
    # ② vs ③: does CIM1 add value on top of depth?
    # ③ vs ⑥: does FIM add value at all?
    # ① vs ④: does FIM improve CIM1 alone?
    # ④ vs ⑥: is depth redundant once you have CIM1 + FIM?

    #REBALNACING EXTENSIONS
    "Imbalance_squared_temporal_MP2": ["rebalancing_imbalance", "squared_starvation_penalty", "squared_congestion_penalty", "gross_starvation_risk", "gross_congestion_risk", "global_onsite_backlog"], # CIM1 + FIM1/FIM2 + MP2: global mass + gross departure/arrival pressure + global onsite backlog 
    "Imbalance_squared_temporal_MP3": ["rebalancing_imbalance", "squared_starvation_penalty", "squared_congestion_penalty", "gross_starvation_risk", "gross_congestion_risk", "global_depot_backlog"], # CIM1 + FIM1/FIM2 + MP3: global mass + gross departure/arrival pressure + global depot backlog
    "Imbalance_squared_temporal_MP1": ["rebalancing_imbalance", "squared_starvation_penalty", "squared_congestion_penalty", "gross_starvation_risk", "gross_congestion_risk", "trailer_cannibalization"], # CIM1 + FIM1/FIM2 + MP1: global mass + gross departure/arrival pressure + trailer cannibalization ratio
    "Imbalance_squared_temporal_MP6": ["rebalancing_imbalance", "squared_starvation_penalty", "squared_congestion_penalty", "gross_starvation_risk", "gross_congestion_risk", "depot_idle_fraction"], # CIM1 + FIM1/FIM2 + MP6: global mass + gross departure/arrival pressure + depot idle fraction
    "Imbalance_squared_temporal_fullMP" : ["rebalancing_imbalance", "squared_starvation_penalty", "squared_congestion_penalty", "gross_starvation_risk", "gross_congestion_risk", "trailer_cannibalization", "global_onsite_backlog", "global_depot_backlog", "depot_idle_fraction"], # CIM1 + FIM1/FIM2 + all MP features 
    "Imbalance_squared_temporal_MP326" : ["rebalancing_imbalance", "squared_starvation_penalty", "squared_congestion_penalty", "gross_starvation_risk", "gross_congestion_risk", "global_onsite_backlog", "global_depot_backlog", "depot_idle_fraction"], # CIM1 + FIM1/FIM2 + MP2/MP3/MP6: global mass + gross departure/arrival pressure + global onsite/depot backlog + depot idle fraction

    "Imbalance_squared_temporal_healthMP" : [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "gross_starvation_risk",
        "gross_congestion_risk",
        "fleet_health_deficit",
        "depot_bound_health_deficit",
        "onsite_health_deficit",
        "maintenance_restoration_value",
    ],
    "Imbalance_squared_temporal_MP3_dest_local" : [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "gross_starvation_risk",
        "gross_congestion_risk",
        "global_depot_backlog",
        "destination_onsite_fraction",
        "destination_depot_fraction",
    ],
    
    "Imbalance_squared_temporal_MP3_full_dest" : [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "gross_starvation_risk",
        "gross_congestion_risk",
        "global_depot_backlog",
        "destination_onsite_fraction",
        "destination_depot_fraction",
        "destination_starv_ratio",
        "destination_cong_ratio",
    ],
    "Imbalance_squared_temporal_destination_station_aware": ["rebalancing_imbalance","squared_starvation_penalty","squared_congestion_penalty", "gross_starvation_risk","gross_congestion_risk", "destination_onsite_fraction","destination_depot_fraction",], # CIM1 + FIM1/FIM2 + destination-local station health/imbalance features


    "Maintenance_full_test": [
        "rebalancing_imbalance",          # CIM1: total L1 imbalance across the network
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "gross_starvation_risk",
        "gross_congestion_risk",
        #"trailer_cannibalization",           # MP1
        #"global_onsite_backlog",             # MP2
        #"global_depot_backlog",              # MP2b — unweighted depot backlog
        #"demand_weighted_depot_backlog",     # MP3 — demand-weighted (fixed)
        #"demand_weighted_onsite_backlog",    # MP4 — new
        #"fleet_broken_fraction",             # MP5 — 0=good, positive=bad
        #"undistributed_depot_inventory",          # MP6 — 0=good, positive=bad
        # Pillar 5: Destination-local (routing discrimination)
        #"destination_starv_ratio",
        #"destination_cong_ratio",
        #"destination_onsite_fraction",
        #"destination_travel_penalty",
        #"cur_station_onsite_fraction",
        #"cur_station_func_deficit",
        #"functional_load_late_pressure",
        #"depot_load_late_pressure",
        "late_depot_return",
    ],

    # Same as Maintenance_full_test but WITHOUT destination features — isolates their impact
    "Maintenance_full_test_no_dest": [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "gross_starvation_risk",
        "gross_congestion_risk",
        "global_onsite_backlog",
        "global_depot_backlog",
        "undistributed_depot_inventory",
        "recoverable_starvation",
    ],
    
    # --- Check 12: Minimal debug experiment matrix ---
    # Run these SHORT (50–100 ep) before full runs to isolate root cause.
    # A/B/C share same features — toggle use_td_lambda + batch_size manually in train_vfa.py:
    #   Debug_A: use_td_lambda=False, batch_size=1,  alpha=0.1  (TD(0) mean batch)
    #   Debug_B: use_td_lambda=True,  batch_size=1,  alpha=0.1  (TD(λ) online)
    #   Debug_C: use_td_lambda=True,  batch_size=1,  alpha=0.01 (TD(λ) tiny alpha)
    "Debug_D_no_maintenance": [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
    ],

    "Debug_E_maint_TD0": [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "global_onsite_backlog",
        "undistributed_depot_inventory",
    ],

    
    # End-of-shift awareness: temporal base + depot feasibility and late-shift route/cargo interactions
    "Imbalance_Squared_Temporal_ShiftAware": [
        "rebalancing_imbalance",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "gross_starvation_risk",
        "gross_congestion_risk",
        "depot_slack_fraction",
        #"can_return_to_depot",
        "depot_return_urgency",
        #"functional_load_late_pressure",
        "depot_load_late_pressure",
        #"non_depot_late_load_pressure",
        "late_depot_return",
    ],

    
    "Imbalanced_Starvation" : ["rebalancing_imbalance", "squared_starvation_penalty"],

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
    # SCRAPPED because of correlation with squared_congestion
    # "Exponential_Temporal": [
    #     "exponential_starvation_penalty", # CIM4
    #     "exponential_congestion_penalty", # CIM5
    #     "gross_starvation_risk",          # FIM1
    #     "gross_congestion_risk",          # FIM2
    # ],

    # -- Gross_And_Net  ------------------------------------------------------
    # Experiment using all FIM features simultaneously.
    # [sq_starvation, sq_congestion, FIM1, FIM2, FIM3, FIM4]
    "Gross_And_Net": [
        "squared_starvation_penalty",     # CIM2: mean squared starvation ratio
        "squared_congestion_penalty",     # CIM3: mean squared congestion ratio
        "gross_starvation_risk",          # FIM1: gross departure pressure
        "gross_congestion_risk",          # FIM2: gross arrival pressure
        "net_starvation_shortfall",       # FIM3: net outflow pressure with demand uncertainty
        "net_congestion_shortfall",       # FIM4: net inflow pressure with demand uncertainty
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
 
def run_all_experiments(seeds: List[int], episodes: int = 200, run_only: Optional[List[str]] = None, alphas: Optional[List[float]] = None, output_dir: str = "results", weight_starvation: float = -1.0, weight_congestion: float = -1.0, weight_fleet_degradation: float = -1.0, weight_trip_served: float = 0.0, gamma: float = 0.99, not_at_depot_at_end_penalty: float = 0.0, functional_bikes_at_end_penalty: float = 0.0, epsilon_start: float = EPSILON_START, epsilon_end: float = EPSILON_END, use_bias_feature: bool = False, use_reward_centering: bool = False, reward_centering_beta: float = 0.01, use_terminal_update: bool = False, use_batch_td_clip: bool = False, batch_td_clip_value: float = 10.0, use_online_td_updates: bool = False, transition_update_interval: int = 0, use_feature_scale_diagnostics: bool = False, diagnostic_every_n_episodes: int = 10, td_lambda: float = 0.0, initial_bias: float | None = None, use_feature_centering: bool = False, feature_centering_beta: float = 0.01, log_candidate_diagnostics: bool = False, log_greedy_comparison: bool = False):
    if run_only:
        unknown = set(run_only) - set(EXPERIMENTS)
        if unknown:
            raise ValueError(f"Unknown experiment(s): {unknown}. Valid: {list(EXPERIMENTS)}")
 
    if alphas is None:
        alphas = [ALPHA_START]
 
    experiments = {k: v for k, v in EXPERIMENTS.items() if run_only is None or k in run_only}
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_parts = []
    if use_bias_feature:
        config_parts.append("bias")
    if use_reward_centering:
        config_parts.append(f"rc{reward_centering_beta:g}")
    if use_terminal_update:
        config_parts.append("term")
    if td_lambda > 0.0:
        config_parts.append(f"lam{td_lambda:g}")
    if use_batch_td_clip:
        config_parts.append(f"clip{batch_td_clip_value:g}")
    if use_online_td_updates:
        config_parts.append("online")
    if transition_update_interval > 0:
        config_parts.append(f"trans{transition_update_interval}")
    if use_feature_scale_diagnostics:
        config_parts.append(f"fsdiag{diagnostic_every_n_episodes}")
    if gamma != 0.99:
        config_parts.append(f"g{gamma:g}")
    if epsilon_start == 0.0 and epsilon_end == 0.0:
        config_parts.append("eps0")
    elif epsilon_start != EPSILON_START or epsilon_end != EPSILON_END:
        config_parts.append(f"eps{epsilon_start:g}-{epsilon_end:g}")
    if initial_bias not in (None, 0.0):
        config_parts.append(f"initb{str(initial_bias).replace('-', 'm').replace('.', 'p')}")
    if use_feature_centering:
        config_parts.append(f"fcenter{feature_centering_beta:g}")
    if log_candidate_diagnostics:
        config_parts.append("canddiag")
    if log_greedy_comparison:
        config_parts.append("gcmp")
    config_suffix = ("_" + "_".join(config_parts)) if config_parts else ""
 
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
                exp_dir = base_dir / f"{exp_name}_alpha_{alpha}{config_suffix}_{timestamp}"
                
                # Safely create the whole chain (models -> custom_name -> exp_name)
                # exist_ok=True ensures subsequent seeds peacefully reuse this folder
                exp_dir.mkdir(parents=True, exist_ok=True)
 
                train(
                    num_episodes=episodes,
                    save_path=exp_dir / f"vfa_{exp_name}_seed{seed_offset}{config_suffix}.pkl",
                    seed_offset=seed_offset,
                    active_features=features,
                    alpha_start=alpha,
                    epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end,
                    gamma=gamma,
                    weight_starvation=weight_starvation,
                    weight_congestion=weight_congestion,
                    weight_fleet_degradation=weight_fleet_degradation,
                    weight_trip_served=weight_trip_served,
                    not_at_depot_at_end_penalty=not_at_depot_at_end_penalty,
                    functional_bikes_at_end_penalty=functional_bikes_at_end_penalty,
                    include_bias=use_bias_feature,
                    use_reward_centering=use_reward_centering,
                    reward_centering_beta=reward_centering_beta,
                    use_terminal_update=use_terminal_update,
                    use_batch_td_clip=use_batch_td_clip,
                    batch_td_clip_value=batch_td_clip_value,
                    use_online_td_updates=use_online_td_updates,
                    transition_update_interval=transition_update_interval,
                    use_feature_scale_diagnostics=use_feature_scale_diagnostics,
                    diagnostic_every_n_episodes=diagnostic_every_n_episodes,
                    td_lambda=td_lambda,
                    initial_bias=initial_bias,
                    use_feature_centering=use_feature_centering,
                    feature_centering_beta=feature_centering_beta,
                    log_candidate_diagnostics=log_candidate_diagnostics,
                    log_greedy_comparison=log_greedy_comparison,
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
        "--episodes", type=int, default=300,
        help="Training episodes per run",
    )
    parser.add_argument(
        "--experiments", nargs="+", type=str, default=None, metavar="NAME",
        help="Subset of experiments to run (default: all)",
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.05], metavar="ALPHA",
        help="List of alpha (learning rate) values to test. e.g. --alphas 0.001 0.005 0.01",
    )

    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save experiment results"
    )
    parser.add_argument(
        "--weight_starvation", type=float, default= -1.0,
        help="Reward weight for starvation events (default: -1.0)",
    )
    parser.add_argument(
        "--weight_congestion", type=float, default= -1.0,
        help="Reward weight for congestion events (default: -1.0)",
    )
    parser.add_argument(
        "--weight_fleet_degradation", type=float, default=-0.0,
        help="Reward weight for fleet degradation/maintenance penalty (default: 0.0)",
    )
    parser.add_argument(
        "--weight_trip_served", type=float, default=0.0,
        help="Positive reward per successful trip served (default: 0.0)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor (default: 0.99 per hour)",
    )
    parser.add_argument(
        "--not_at_depot_at_end_penalty", type=float, default=0.0,
        help="Per-step penalty (ramped) for being away from depot near shift end (default: 0.0, suggested: -2.0)",
    )
    parser.add_argument(
        "--functional_bikes_at_end_penalty", type=float, default=0.0,
        help="Per-bike per-step penalty for functional cargo near shift end (default: 0.0, suggested: -0.2)",
    )
    parser.add_argument(
        "--epsilon_start",
        type=float,
        default=EPSILON_START,
        help="Initial epsilon for epsilon-greedy exploration.",
    )
    parser.add_argument(
        "--epsilon_end",
        type=float,
        default=EPSILON_END,
        help="Final epsilon for epsilon-greedy exploration.",
    )
    parser.add_argument(
        "--use_bias_feature",
        action="store_true",
        help="Add a constant bias/intercept feature to the linear VFA",
    )
    parser.add_argument(
        "--use_reward_centering",
        action="store_true",
        help="Subtract a running reward mean before TD updates",
    )
    parser.add_argument(
        "--reward_centering_beta",
        type=float,
        default=0.01,
        help="EMA step size for reward centering baseline (default: 0.01)",
    )
    parser.add_argument(
        "--use_terminal_update",
        action="store_true",
        help="Append terminal transition with zero bootstrap at the end of each episode",
    )
    parser.add_argument(
        "--use_batch_td_clip",
        action="store_true",
        help="Clip TD errors inside the episode batch update",
    )
    parser.add_argument(
        "--batch_td_clip_value",
        type=float,
        default=10.0,
        help="Absolute TD error clip used when --use_batch_td_clip is set",
    )
    parser.add_argument(
        "--use_online_td_updates",
        action="store_true",
        help="Apply TD updates immediately at every transition instead of episode-batch updates.",
    )
    parser.add_argument(
        "--transition_update_interval",
        type=int,
        default=0,
        help="Apply one mean-gradient TD update every N buffered transitions. 0 keeps episode-batch updates.",
    )
    parser.add_argument(
        "--use_feature_scale_diagnostics",
        action="store_true",
        help="Print per-feature scale diagnostics during batch updates",
    )
    parser.add_argument(
        "--diagnostic_every_n_episodes",
        type=int,
        default=10,
        help="Frequency for heavy diagnostics such as feature scale reports",
    )
    parser.add_argument(
        "--td_lambda",
        type=float,
        default=0.0,
        help="Eligibility trace lambda. Use 0.0 for TD(0); e.g. 0.3 for low TD(lambda)",
    )
    parser.add_argument(
        "--initial_bias",
        type=float,
        default=None,
        help="Initial value for the bias/intercept weight. Requires --use_bias_feature.",
    )
    parser.add_argument(
        "--use_feature_centering",
        action="store_true",
        help="Center non-bias VFA features with a running candidate-set mean before value/TD updates.",
    )
    parser.add_argument(
        "--feature_centering_beta",
        type=float,
        default=0.01,
        help="EMA step size for feature centering when --use_feature_centering is set.",
    )
    parser.add_argument(
        "--log_candidate_diagnostics",
        action="store_true",
        help="Write per-decision candidate value spread diagnostics to CSV.",
    )
    parser.add_argument(
        "--log_greedy_comparison",
        action="store_true",
        help="Write VFA-vs-greedy-maintenance decision comparison diagnostics to CSV.",
    )

    args = parser.parse_args()
    if args.epsilon_start < 0.0 or args.epsilon_end < 0.0:
        raise ValueError(
            f"--epsilon_start and --epsilon_end must be non-negative, "
            f"got {args.epsilon_start}, {args.epsilon_end}"
        )
    if args.transition_update_interval < 0:
        raise ValueError(f"--transition_update_interval must be >= 0, got {args.transition_update_interval}")
    if args.use_online_td_updates and args.transition_update_interval > 0:
        raise ValueError("--use_online_td_updates and --transition_update_interval are mutually exclusive")
    if not 0.0 <= args.td_lambda <= 1.0:
        raise ValueError(f"--td_lambda must be in [0, 1], got {args.td_lambda}")
    if not 0.0 <= args.feature_centering_beta <= 1.0:
        raise ValueError(f"--feature_centering_beta must be in [0, 1], got {args.feature_centering_beta}")

    run_all_experiments(
        seeds=args.seeds,
        episodes=args.episodes,
        run_only=args.experiments,
        alphas=args.alphas,
        output_dir=args.output_dir,
        weight_starvation=args.weight_starvation,
        weight_congestion=args.weight_congestion,
        weight_fleet_degradation=args.weight_fleet_degradation,
        weight_trip_served=args.weight_trip_served,
        gamma=args.gamma,
        not_at_depot_at_end_penalty=args.not_at_depot_at_end_penalty,
        functional_bikes_at_end_penalty=args.functional_bikes_at_end_penalty,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        use_bias_feature=args.use_bias_feature,
        use_reward_centering=args.use_reward_centering,
        reward_centering_beta=args.reward_centering_beta,
        use_terminal_update=args.use_terminal_update,
        use_batch_td_clip=args.use_batch_td_clip,
        batch_td_clip_value=args.batch_td_clip_value,
        use_online_td_updates=args.use_online_td_updates,
        transition_update_interval=args.transition_update_interval,
        use_feature_scale_diagnostics=args.use_feature_scale_diagnostics,
        diagnostic_every_n_episodes=args.diagnostic_every_n_episodes,
        td_lambda=args.td_lambda,
        initial_bias=args.initial_bias,
        use_feature_centering=args.use_feature_centering,
        feature_centering_beta=args.feature_centering_beta,
        log_candidate_diagnostics=args.log_candidate_diagnostics,
        log_greedy_comparison=args.log_greedy_comparison,
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
 
 
