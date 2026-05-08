"""
vfa_features.py -  Feature Vector φ(S^x) for the Time-Indexed Linear VFA

This is the CANONICAL definition of every feature used in the VFA.
To change the feature set:
  1. Edit get_feature_names().
  2. Edit the corresponding computation block in extract().
  3. Update the return array in extract() to match.
  4. Retrain - nothing else needs touching.

LinearVFAPolicy imports this module as the single source of truth via:
    - get_feature_names()
    - extract(...)
    - as_dict(...)

Features are organised around four Operational Pillars:

  Pillar 1  Current System Imbalance  (CIM) — always active
  Pillar 2  Future System Imbalance   (FIM) — demand_horizon_enabled
  Pillar 3  Maintenance Pressure      (MP)  — maintenance_enabled
  Pillar 4  Spatial & Logistic        (SLC) — logistics_enabled

──────────────────────────────────────────────────────────────────────────────
Notation
────────
  I_i       functional bikes at station i  (post-decision)
  T_i       time-indexed target inventory at station i
  C_i       physical rack capacity of station i
  λ_i^net   net demand at station i over the rolling horizon  (leave − arrive)
  λ_i^out   gross expected departures at station i over the rolling horizon
  λ_i^in    gross expected arrivals at station i over the rolling horizon
  free_i    free docks at station i  (C_i − I_i − broken_i)
  q_func    functional bikes on the vehicle (post-decision)
  q_depot   broken bikes on the vehicle (post-decision)
  K         vehicle capacity
  d_i       travel time from vehicle location to station i (minutes)
  t_rem     time remaining in shift (minutes)
  L         total shift length (minutes)
  F         total fleet size (bikes)
  N         number of stations

──────────────────────────────────────────────────────────────────────────────
Pillar 1  —  Current System Imbalance  (CIM, always active)
──────────────────────────────────────────────────────────────────────────────
  CIM1  rebalancing_imbalance          Σ_i |I_i - T_i| / (0.5 * Σ_i C_i)
  CIM2  squared_starvation_penalty     (1/N) Σ_i (max(0, T_i-I_i) / T_i)^2
  CIM3  squared_congestion_penalty     (1/N) Σ_i (max(0, I_i-T_i) / (C_i-T_i))^2
  CIM4  exponential_starvation_penalty (1/N) Σ_i (exp(3 * starv_ratio_i) - 1) / (exp(3) - 1)
  CIM5  exponential_congestion_penalty (1/N) Σ_i (exp(3 * cong_ratio_i) - 1) / (exp(3) - 1)
  CIM6  starvation_severity_max        Q95 of starvation ratio across stations
  CIM7  congestion_severity_max        Q95 of congestion ratio across stations
  CIM8  starvation_count               fraction of stations with starvation ratio ≥ 0.9
  CIM9 congestion_count               fraction of stations with congestion ratio ≥ 0.9

──────────────────────────────────────────────────────────────────────────────
Pillar 2  —  Future System Imbalance  (FIM, appended if demand_horizon_enabled)
──────────────────────────────────────────────────────────────────────────────
  FIM1  gross_starvation_risk          (1/N) Σ_i [λ_i^out + √λ_i^out − I_i]^+ / T_i
  FIM2  gross_congestion_risk          (1/N) Σ_i [λ_i^in + √λ_i^in − free_i]^+ / (C_i − T_i)
  FIM3  net_starvation_shortfall       (1/N) Σ_i [λ_i^out − λ_i^in + √(λ_i^out+λ_i^in)]^+ / T_i
  FIM4  net_congestion_shortfall       (1/N) Σ_i [λ_i^in − λ_i^out + √(λ_i^out+λ_i^in)]^+ / (C_i − T_i)

──────────────────────────────────────────────────────────────────────────────
Pillar 3  —  Maintenance Pressure  (MP, appended if maintenance_enabled)
──────────────────────────────────────────────────────────────────────────────
  MP1  trailer_cannibalization         q_depot / K
  MP2  global_onsite_backlog           Σ_i onsite_i / F
  MP3  demand_weighted_depot_backlog   Σ_i (depot_i * λ_i^out) / (Λ_max * F)  [FIXED]
  MP4  demand_weighted_onsite_backlog  Σ_i (onsite_i * λ_i^out) / (Λ_max * F) [NEW]
  MP5  fleet_broken_fraction           (Σ onsite + Σ depot + in_repair) / F     [NEW — 0=good]
  MP6  undistributed_depot_inventory   -depot.fixed_queue / (0.07F)             [fixed-queue pressure]
  MP7  depot_idle_fraction             depot.fixed_queue / F                    [NEW — 0=good]
  MP8  recoverable_starvation          Σ_i onsite_i * 1[func_i < T_i] / F      [NEW]
  MP9  maintenance_urgency             MP2 * CIM8  (RESTORED)
  MP11 fleet_failure_risk              mean bike-level component failure risk [0=good]
  MP12 fleet_health_deficit            1 - mean bike health                 [0=good]
  MP13 fleet_low_health_fraction       fraction of bikes below health floor [0=good]
  MP14 depot_bound_health_deficit      depot-bound bad health pressure      [0=good]
  MP15 onsite_health_deficit           onsite bad health pressure           [0=good]
  MP16 maintenance_restoration_value   candidate maintenance value          [0=neutral, high=good]

──────────────────────────────────────────────────────────────────────────────
Pillar 4  —  Spatial & Logistic Constraints  (SLC, appended if logistics_enabled)
──────────────────────────────────────────────────────────────────────────────
  SLC1  time_remaining_fraction        t_rem / L
  SLC2  functional_bikes_time_penalty  (q_func / K) * (1 - SLC1)
  SLC3  reachable_imbalance_fraction   Σ_i |I_i-T_i|*1[d_i≤t_rem] / Σ_i |I_i-T_i|
  SLC4  recoverable_imbalance_fraction fraction of imbalance reachable and serviceable now
  SLC5  imbalance_hotspot_distance     distance to the worst-imbalance hotspot, normalised
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
from settings import SERVICE_TIME_FROM, SERVICE_TIME_TO


# ─────────────────────────────────────────────────────────────────────────────
# Feature registry and feature-selection metadata
# ─────────────────────────────────────────────────────────────────────────────

CIM_FEATURE_NAMES = [
    "rebalancing_imbalance",          # CIM1
    "squared_starvation_penalty",     # CIM2
    "squared_congestion_penalty",     # CIM3
    "exponential_starvation_penalty", # CIM4
    "exponential_congestion_penalty", # CIM5
    "starvation_severity_max",        # CIM6
    "congestion_severity_max",        # CIM7
    "starvation_count",               # CIM8
    "congestion_count",               # CIM9
]

FIM_FEATURE_NAMES = [
    "gross_starvation_risk",          # FIM1
    "gross_congestion_risk",          # FIM2
    "net_starvation_shortfall",       # FIM3
    "net_congestion_shortfall",       # FIM4
]

REBALANCING_CORRELATION_FEATURES = CIM_FEATURE_NAMES + FIM_FEATURE_NAMES


CIM_FEATURE_NAMES = [
    "rebalancing_imbalance",          # CIM1
    "squared_starvation_penalty",     # CIM2
    "squared_congestion_penalty",     # CIM3
    "exponential_starvation_penalty", # CIM4
    "exponential_congestion_penalty", # CIM5
    "starvation_severity_max",        # CIM6
    "congestion_severity_max",        # CIM7
    "starvation_count",               # CIM8
    "congestion_count",               # CIM9
]
FIM_FEATURE_NAMES = [
    "gross_starvation_risk",          # FIM1
    "gross_congestion_risk",          # FIM2
    "net_starvation_shortfall",       # FIM3
    "net_congestion_shortfall",       # FIM4
]

REBALANCING_CORRELATION_FEATURES = CIM_FEATURE_NAMES + FIM_FEATURE_NAMES


BASE_REBALANCING_FEATURES = [
    "rebalancing_imbalance",
    "squared_starvation_penalty",
    "squared_congestion_penalty",
    "gross_starvation_risk",
    "gross_congestion_risk",
]

MAINTENANCE_FEATURE_POOL = [
    # Existing broad maintenance descriptors.
    "global_onsite_backlog",
    "global_depot_backlog",
    "demand_weighted_onsite_backlog",
    "demand_weighted_depot_backlog",
    "fleet_broken_fraction",
    "depot_idle_fraction",
    "fleet_failure_risk",
    "fleet_low_health_fraction",
    "maintenance_restoration_value",

    # Mechanism-specific candidate features for screening.
    "onsite_shortage_pressure",
    "depot_shortage_pressure",
    "current_station_onsite_shortage_pressure",
    "current_station_depot_shortage_pressure",
    "broken_cargo_distance_pressure",
    "late_broken_cargo_pressure",
    "repaired_idle_away_pressure",
    "depot_pipeline_pressure",
    "depot_queue_pickup_opportunity",
    "degradation_demand_exposure",
]

FEATURE_METADATA: Dict[str, Dict[str, object]] = {
    "global_onsite_backlog": {
        "family": "onsite_backlog",
        "high_means": "many on-site broken bikes remain in the system",
        "expected_sign": "negative",
        "priority": "medium",
    },
    "global_depot_backlog": {
        "family": "depot_backlog",
        "high_means": "many depot-bound broken bikes remain at stations",
        "expected_sign": "negative",
        "priority": "medium",
    },
    "demand_weighted_onsite_backlog": {
        "family": "demand_weighted_backlog",
        "high_means": "on-site broken bikes are located at high-departure stations",
        "expected_sign": "negative",
        "priority": "high",
    },
    "demand_weighted_depot_backlog": {
        "family": "demand_weighted_backlog",
        "high_means": "depot-bound broken bikes are located at high-departure stations",
        "expected_sign": "negative",
        "priority": "high",
    },
    "fleet_broken_fraction": {
        "family": "fleet_availability",
        "high_means": "a large share of the fleet is unavailable due to maintenance",
        "expected_sign": "negative",
        "priority": "medium",
    },
    "depot_idle_fraction": {
        "family": "depot_pipeline",
        "high_means": "many repaired bikes are waiting idle at the depot",
        "expected_sign": "negative",
        "priority": "medium",
    },
    "fleet_failure_risk": {
        "family": "degradation",
        "high_means": "the functional fleet has high expected component failure risk",
        "expected_sign": "negative",
        "priority": "medium",
    },
    "fleet_low_health_fraction": {
        "family": "degradation",
        "high_means": "many bikes are below the low-health threshold",
        "expected_sign": "negative",
        "priority": "medium",
    },
    "maintenance_restoration_value": {
        "family": "immediate_repair_opportunity",
        "high_means": "the candidate action restores valuable broken bikes",
        "expected_sign": "positive",
        "priority": "high",
    },
    "onsite_shortage_pressure": {
        "family": "immediate_repair_opportunity",
        "high_means": "on-site broken bikes remain at shortage-prone, high-demand stations",
        "expected_sign": "negative",
        "priority": "high",
    },
    "depot_shortage_pressure": {
        "family": "depot_backlog",
        "high_means": "depot-bound broken bikes remain at shortage-prone, high-demand stations",
        "expected_sign": "negative",
        "priority": "high",
    },
    "current_station_onsite_shortage_pressure": {
        "family": "immediate_repair_opportunity",
        "high_means": "the current station still has unrepaired on-site bikes and shortage pressure",
        "expected_sign": "negative",
        "priority": "high",
    },
    "current_station_depot_shortage_pressure": {
        "family": "depot_removal_opportunity",
        "high_means": "the current station still has depot-bound bikes and shortage pressure",
        "expected_sign": "negative",
        "priority": "medium",
    },
    "broken_cargo_distance_pressure": {
        "family": "vehicle_logistics",
        "high_means": "the vehicle is carrying broken bikes far from the depot",
        "expected_sign": "negative",
        "priority": "high",
    },
    "late_broken_cargo_pressure": {
        "family": "vehicle_logistics",
        "high_means": "the vehicle carries broken bikes late in the shift while not routing to the depot",
        "expected_sign": "negative",
        "priority": "high",
    },
    "repaired_idle_away_pressure": {
        "family": "depot_pipeline",
        "high_means": "repaired bikes wait at depot while the candidate route avoids depot",
        "expected_sign": "negative",
        "priority": "high",
    },
    "depot_pipeline_pressure": {
        "family": "depot_pipeline",
        "high_means": "many bikes are unavailable in the depot pipeline",
        "expected_sign": "negative",
        "priority": "medium",
    },
    "depot_queue_pickup_opportunity": {
        "family": "depot_pipeline",
        "high_means": "the candidate route can pick up repaired depot bikes with available vehicle space",
        "expected_sign": "positive",
        "priority": "medium",
    },
    "degradation_demand_exposure": {
        "family": "degradation",
        "high_means": "high fleet failure risk coincides with future starvation pressure",
        "expected_sign": "negative",
        "priority": "low",
    },
}


def get_base_rebalancing_feature_names() -> List[str]:
    """Compact base feature set used as the benchmark in maintenance tests."""
    return list(BASE_REBALANCING_FEATURES)


def get_maintenance_feature_pool_names() -> List[str]:
    """Candidate maintenance feature pool for pre-training screening."""
    return list(MAINTENANCE_FEATURE_POOL)


def get_feature_metadata() -> Dict[str, Dict[str, object]]:
    """Interpretability metadata used by screening scripts and thesis tables."""
    return {name: dict(meta) for name, meta in FEATURE_METADATA.items()}

def get_feature_names(
    maintenance_enabled: bool = False,
    logistics_enabled: bool = False,
    demand_horizon_enabled: bool = True,
    destination_features_enabled: bool = True,
    include_bias: bool = False,
) -> list:
    """Returns the canonical feature name list for the active operational pillars.

    Pillar 1 (Current System Imbalance) is always active.
    Pillar 2 (Future System Imbalance)   — demand_horizon_enabled
    Pillar 3 (Asset Health & Recovery)   — maintenance_enabled
    Pillar 4 (Spatial & Logistic)        — logistics_enabled
    """

    names = ["bias"] if include_bias else []

    # Pillar 1: Current System Imbalance (CIM, always active)
    names.extend(CIM_FEATURE_NAMES)

    # Pillar 2: Future System Imbalance (FIM)
    if demand_horizon_enabled:
        names.extend(FIM_FEATURE_NAMES)

    # Pillar 3: Maintenance Pressure (MP)
    if maintenance_enabled:
        names.extend([
            "trailer_cannibalization",           # MP1
            "global_onsite_backlog",             # MP2
            "global_depot_backlog",              # MP2b
            "demand_weighted_depot_backlog",     # MP3 — demand-weighted (fixed)
            "demand_weighted_onsite_backlog",    # MP4 — new
            "fleet_broken_fraction",             # MP5 — 0=good, positive=bad
            "depot_idle_fraction",               # MP6 — 0=good, positive=bad
            "recoverable_starvation",            # MP8 — new
            "maintenance_urgency",               # MP9 — restored
            "rush_hour_onsite_backlog",          # MP10 — time-aware penalty
            "fleet_failure_risk",                # MP11 — 0=good, positive=bad
            "fleet_health_deficit",              # MP12 — 0=good, positive=bad
            "fleet_low_health_fraction",         # MP13 — 0=good, positive=bad
            "depot_bound_health_deficit",        # MP14 — 0=good, positive=bad
            "onsite_health_deficit",             # MP15 — 0=good, positive=bad
            "maintenance_restoration_value",     # MP16 — 0=neutral, positive=good
            "onsite_shortage_pressure",
            "depot_shortage_pressure",
            "current_station_onsite_shortage_pressure",
            "current_station_depot_shortage_pressure",
            "broken_cargo_distance_pressure",
            "late_broken_cargo_pressure",
            "repaired_idle_away_pressure",
            "depot_pipeline_pressure",
            "depot_queue_pickup_opportunity",
            "degradation_demand_exposure",
        ])

    # Pillar 4: Spatial & Logistic Constraints (SLC)
    if logistics_enabled:
        names.extend([
            "time_remaining_fraction",        # SLC1
            "functional_bikes_time_penalty",  # SLC2
            "reachable_imbalance_fraction",   # SLC3
            "recoverable_imbalance_fraction", # SLC4
            "imbalance_hotspot_distance",     # SLC5
        ])

    # Pillar 5: Destination-Local Features
    if destination_features_enabled:
        names.extend([
            "destination_starv_ratio",        # max(0, T_nxt - I_nxt) / T_nxt
            "destination_cong_ratio",         # max(0, I_nxt - T_nxt) / (C_nxt - T_nxt)
            "destination_onsite_fraction",    # onsite[nxt] / C_nxt
            "destination_depot_fraction",      # depot[nxt] / C_nxt
            "destination_travel_penalty",     # dist_to_next / max_system_travel_time
            "destination_roi_starvation",     # Starvation / Travel Penalty
            "cur_station_onsite_fraction",    # onsite[cur] / C_cur  (post-decision)
            "cur_station_func_deficit",       # max(0, T_cur - I_cur) / C_cur  (post-decision)
            "functional_load_late_pressure",  # (q_func/K) * (1 - shift_remaining)
            "depot_load_late_pressure",       # (q_depot/K) * (1 - shift_remaining)
            "non_depot_late_load_pressure",   # 1[next not depot] * ((q_func+q_depot)/K) * late
            "late_depot_return",              # 1[next is depot] * late
            "depot_slack_fraction",           # (t_rem - d_depot) / L, clamped [-1, 1]
            "can_return_to_depot",            # 1.0 if t_rem > d_depot else 0.0
            "depot_return_urgency",           # d_depot / max(t_rem, 1), clamped [0, 1]
        ])

    return names


#_MORNING_PEAK_HOUR: int = 8  # 08:00 — bike-sharing morning rush anchor


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract(
    func: np.ndarray,
    onsite: np.ndarray,
    depot: np.ndarray,
    target: np.ndarray,
    capacities: np.ndarray,
    leave_activity: np.ndarray,
    arrive_activity: np.ndarray,
    dist_to_stations: np.ndarray,
    func_cargo_veh: float,
    depot_cargo_veh: float,
    vehicle_capacity: int,
    dist_to_depot: float,
    lambda_max_system: float,
    max_gravity: float,
    fleet_size: float,
    total_stations: int,
    depot_in_repair: float = 0.0,
    depot_fixed_queue: float = 0.0,
    fleet_failure_risk: float = 0.0,
    fleet_health_deficit: float = 0.0,
    fleet_low_health_fraction: float = 0.0,
    depot_bound_health_deficit: float = 0.0,
    onsite_health_deficit: float = 0.0,
    maintenance_restoration_value: float = 0.0,
    maintenance_enabled: bool = True,
    logistics_enabled: bool = False,
    time_remaining: Optional[float] = None,
    shift_length: float = 1440.0,
    demand_horizon_enabled: bool = True,
    current_time_minutes: float = 0.0,
    current_day_of_week: int = 0,
    target_matrix: Optional[np.ndarray] = None,  # shape (7, 24, N) — for multi-horizon look-ahead
    horizon_hours: int = 4,
    next_station_idx: int = -1,       # index into func/onsite/target arrays for destination station
    cur_station_idx: int = -1,        # index into func/onsite/target arrays for CURRENT station
    dist_to_next: float = 0.0,        # travel time (minutes) from current location to destination
    max_travel_time: float = 60.0,    # system-wide max travel time for normalisation
    next_is_depot: bool = False,
    destination_features_enabled: bool = False,
    include_bias: bool = False,
) -> np.ndarray:

    features = [1.0] if include_bias else []


    # ── Safe denominators ─────────────────────────────────────────────────────
    K            = max(vehicle_capacity, 1)
    K_safe       = max(float(vehicle_capacity), 1.0)
    N            = len(func)
    F_safe       = max(fleet_size, 1.0)
    lam_max_safe = max(lambda_max_system, 1.0)
    grav_safe    = max(max_gravity, 1.0)

    # ── Shared pre-computations (reused across multiple features) ─────────────
    total_cap_half  = 0.5 * np.sum(capacities)
    total_imbalance = np.sum(np.abs(func - target))

    # ── Intermediate vehicle-state fractions (not features themselves) ────────
    veh_load      = func_cargo_veh / K_safe                          # q_func / K
    veh_free_frac = max(0.0, K_safe - func_cargo_veh - depot_cargo_veh) / K_safe  # free / K

    # ── Demand decomposition — sum gross flows over the horizon ──────────────
    # leave_activity and arrive_activity are either (H, N) or (N,).
    # gross_outflow / gross_inflow are always (N,) expected bike counts.
    if leave_activity.ndim == 2:
        gross_outflow = np.sum(leave_activity,  axis=0)   # Σ_t leave_t  (N,)
        gross_inflow  = np.sum(arrive_activity, axis=0)   # Σ_t arrive_t (N,)
    else:
        gross_outflow = leave_activity
        gross_inflow  = arrive_activity
    net_activity = gross_outflow - gross_inflow            # (N,) used by B3

    # ── Shared per-station starvation/congestion ratios (reused in A3, A9) ───
    target_safe  = np.maximum(1.0, target)
    cap_rem_safe = np.maximum(1.0, capacities - target)
    starv_ratio  = np.maximum(0.0, target - func) / target_safe   # (T_i - I_i) / T_i
    cong_ratio   = np.maximum(0.0, func - target) / cap_rem_safe  # (I_i - T_i) / (C_i - T_i)

    # =========================================================================
    # Pillar 1: Current System Imbalance (CIM)
    # =========================================================================

    # CIM1: Rebalancing Imbalance — total L1 deviation from target, normalised by half capacity
    phi_imbalance = total_imbalance / max(total_cap_half, 1.0)

    # CIM3: Squared Starvation Penalty — mean squared starvation depth
    phi_starvation_sq = np.sum(starv_ratio**2) / N

    # ── Diagnostic: one-shot print if phi_starvation_sq is stuck at zero ──────
    if phi_starvation_sq == 0.0 and not getattr(extract, "_printed_starv_diag", False):
        print(f"\n[DIAGNOSTIC] squared_starvation_penalty = 0.0!")
        print(f"  max target: {np.max(target):.1f}   total func: {np.sum(func):.1f}")
        zero_targets = np.sum(target <= 0.0)
        print(f"  stations with target=0: {zero_targets}/{N}")
        if zero_targets > N * 0.8:
            print("  -> Target algorithm is outputting 0 for almost everything.")
        else:
            print("  -> Van post-decision state solved starvation, or city is over-saturated.")
        extract._printed_starv_diag = True

    # CIM4: Squared Congestion Penalty — mean squared congestion depth
    phi_congestion_sq = np.sum(cong_ratio**2) / N

    # CIM5: Exponential Starvation Penalty — exponential penalty to increase tail response
    phi_starvation_exp = np.sum((np.exp(3.0 * starv_ratio) - 1.0) / (np.exp(3.0) - 1.0)) / N

    # CIM6: Exponential Congestion Penalty
    phi_congestion_exp = np.sum((np.exp(3.0 * cong_ratio) - 1.0) / (np.exp(3.0) - 1.0)) / N

    # CIM6: Starvation Severity (Q95) — 95th-percentile starvation ratio
    phi_starvation_max = float(np.quantile(starv_ratio, 0.95))

    # CIM7: Congestion Severity (Q95) — 95th-percentile congestion ratio
    phi_congestion_max = float(np.quantile(cong_ratio, 0.95))

    # CIMx: Starvation Variance — spread of the starvation problem
    #phi_starv_var = float(np.var(starv_ratio))

    # CIM8: Severe Station Starvation Count — fraction with starvation ratio >= 0.9
    phi_starvation_cnt = float(np.sum(starv_ratio >= 0.9)) / N

    # CIM9: Severe Station Congestion Count — fraction with congestion ratio >= 0.9
    phi_congestion_cnt = float(np.sum(cong_ratio >= 0.9)) / N

    cat_a = [
        phi_imbalance, phi_starvation_sq, phi_congestion_sq,
        phi_starvation_exp, phi_congestion_exp,
        phi_starvation_max, phi_congestion_max,
        phi_starvation_cnt, phi_congestion_cnt
    ]
    features.extend(cat_a)

    # =========================================================================
    # Pillar 2: Future System Imbalance (FIM)
    # =========================================================================
    if demand_horizon_enabled:
        # FIM1: Gross Starvation Risk
        starvation_shortfall  = np.maximum(0.0, gross_outflow + np.sqrt(gross_outflow) - func)
        phi_gross_starv       = float(np.mean(starvation_shortfall / target_safe))

        # FIM2: Gross Congestion Risk
        free_docks_d          = np.maximum(0.0, capacities - func - onsite)
        congestion_shortfall  = np.maximum(0.0, gross_inflow + np.sqrt(gross_inflow) - free_docks_d)
        phi_gross_cong        = float(np.mean(congestion_shortfall / cap_rem_safe))

        # FIM3: Net Starvation Shortfall — net outflow pressure vs current inventory
        net_std_dev = np.sqrt(gross_outflow + gross_inflow)
        phi_net_starv_shortfall = float(np.mean(np.maximum(0.0, net_activity + net_std_dev - func) / target_safe))

        # FIM4: Net Congestion Shortfall — net inflow pressure vs current free docks
        phi_net_cong_shortfall  = float(np.mean(np.maximum(0.0, -net_activity + net_std_dev - free_docks_d) / cap_rem_safe))

        cat_d = [phi_gross_starv, phi_gross_cong, phi_net_starv_shortfall, phi_net_cong_shortfall]
        features.extend(cat_d)

    # =========================================================================
    # Pillar 3: Maintenance Pressure (MP)
    # =========================================================================
    if maintenance_enabled:
        dw_denom = max(lam_max_safe * F_safe * 0.30, 1.0)
        demand_weight = gross_outflow / max(lam_max_safe, 1.0)
        demand_weight = np.clip(demand_weight, 0.0, 1.0)
        shortage_demand_weight = np.maximum(starv_ratio, demand_weight)
        max_dist_safe = max(float(np.max(dist_to_stations)) if len(dist_to_stations) else 0.0,
                            float(dist_to_depot), max_travel_time, 1.0)

        # MP1: Trailer Cannibalization
        phi_cannibalization = float(depot_cargo_veh) / K

        # MP2: Global Onsite Backlog — normalised by expected max onsite broken (≈15% of fleet)
        phi_onsite_backlog = float(np.sum(onsite)) / (F_safe * 0.30)

        # MP2b: Global Depot Backlog — normalised by expected max depot broken (≈15% of fleet)
        phi_depot_backlog = float(np.sum(depot)) / (F_safe * 0.30)

        # MP3: Demand-Weighted Depot Backlog (FIXED)
        phi_dw_depot_backlog = float(np.dot(depot, gross_outflow)) / dw_denom

        # MP4: Demand-Weighted Onsite Backlog
        phi_dw_onsite_backlog = float(np.dot(onsite, gross_outflow)) / dw_denom

        # MP5: Fleet Broken Fraction — normalised by expected max total broken (≈20% of fleet)
        phi_fleet_broken = (float(np.sum(onsite)) + float(np.sum(depot)) + depot_in_repair) / (F_safe * 0.20)

        # MP6: Depot Idle Fraction — bikes repaired but not yet picked up.
        # Negative: more bikes sitting idle at depot = worse state (capacity wasted).
        # Normalised by fleet (raw fraction), not 3% cap — 3% was too tight and caused divergence.
        phi_depot_idle = depot_fixed_queue / (F_safe*0.30)
        
        # MP8: Recoverable Starvation — normalised by expected max recoverable (≈10% of fleet)
        is_starving = (func < target).astype(np.float64)
        phi_rec_starvation = float(np.dot(is_starving, onsite)) / (F_safe * 0.10)

        # MP9: Maintenance Urgency — onsite backlog × starving station count
        phi_maint_urgency = phi_onsite_backlog * phi_starvation_cnt

        # MP10: Rush Hour Onsite Backlog
        # Is it during peak hours where immediate functional bikes are needed? (e.g. 07:00-09:00 or 15:00-17:00)
        hour_of_day = (current_time_minutes // 60) % 24
        is_rush_hour = 1.0 if (6 <= hour_of_day < 10) or (15 <= hour_of_day < 18) else 0.0
        phi_rush_hour_onsite = is_rush_hour * phi_dw_onsite_backlog

        # Candidate-pool maintenance mechanisms.  All pressure features are
        # directional: 0 is good/no pressure, larger values are worse unless the
        # feature name explicitly says "opportunity".
        phi_onsite_shortage_pressure = float(np.dot(onsite, shortage_demand_weight)) / (F_safe * 0.10)
        phi_depot_shortage_pressure = float(np.dot(depot, shortage_demand_weight)) / (F_safe * 0.10)

        if cur_station_idx >= 0 and cur_station_idx < N:
            cur_cap_safe = float(max(capacities[cur_station_idx], 1.0))
            cur_shortage_pressure = max(starv_ratio[cur_station_idx], demand_weight[cur_station_idx])
            phi_cur_onsite_shortage_pressure = (
                float(onsite[cur_station_idx]) / cur_cap_safe
            ) * float(cur_shortage_pressure)
            phi_cur_depot_shortage_pressure = (
                float(depot[cur_station_idx]) / cur_cap_safe
            ) * float(cur_shortage_pressure)
        else:
            phi_cur_onsite_shortage_pressure = 0.0
            phi_cur_depot_shortage_pressure = 0.0

        _service_window_min = max((SERVICE_TIME_TO - SERVICE_TIME_FROM) * 60.0, 1.0)
        _clock_min = current_time_minutes % 1440.0
        _close_min = SERVICE_TIME_TO * 60.0
        _time_remaining_abs = max(0.0, _close_min - _clock_min)
        phi_shift_remaining_for_mp = min(1.0, _time_remaining_abs / _service_window_min)
        phi_late_for_mp = max(0.0, 1.0 - phi_shift_remaining_for_mp)
        depot_cargo_frac = float(depot_cargo_veh) / K_safe
        phi_broken_cargo_distance_pressure = depot_cargo_frac * (float(dist_to_depot) / max_dist_safe)
        phi_late_broken_cargo_pressure = depot_cargo_frac * phi_late_for_mp * (0.0 if next_is_depot else 1.0)

        phi_repaired_idle_away_pressure = phi_depot_idle * (0.0 if next_is_depot else 1.0)
        phi_depot_pipeline_pressure = (
            float(np.sum(depot)) + float(depot_cargo_veh) + float(depot_in_repair) + float(depot_fixed_queue)
        ) / (F_safe * 0.30)
        phi_depot_queue_pickup_opportunity = phi_depot_idle * veh_free_frac * (1.0 if next_is_depot else 0.0)
        phi_degradation_demand_exposure = float(np.clip(fleet_failure_risk, 0.0, 1.0)) * phi_gross_starv if demand_horizon_enabled else 0.0

        cat_b = [
            phi_cannibalization, phi_onsite_backlog, phi_depot_backlog, phi_dw_depot_backlog,
            phi_dw_onsite_backlog, phi_fleet_broken, phi_depot_idle,
            phi_rec_starvation, phi_maint_urgency,
            phi_rush_hour_onsite,
            float(np.clip(fleet_failure_risk, 0.0, 1.0)),
            float(np.clip(fleet_health_deficit, 0.0, 1.0)),
            float(np.clip(fleet_low_health_fraction, 0.0, 1.0)),
            float(max(0.0, depot_bound_health_deficit)),
            float(max(0.0, onsite_health_deficit)),
            float(max(0.0, maintenance_restoration_value)),
            phi_onsite_shortage_pressure,
            phi_depot_shortage_pressure,
            phi_cur_onsite_shortage_pressure,
            phi_cur_depot_shortage_pressure,
            phi_broken_cargo_distance_pressure,
            phi_late_broken_cargo_pressure,
            phi_repaired_idle_away_pressure,
            phi_depot_pipeline_pressure,
            phi_depot_queue_pickup_opportunity,
            phi_degradation_demand_exposure,
        ]
        features.extend(cat_b)


    # =========================================================================
    # Pillar 4: Spatial & Logistic Constraints (SLC)
    # =========================================================================
    if logistics_enabled:
        if time_remaining is None:
            time_remaining = shift_length
        shift_length_safe = max(shift_length, 1.0)

        #NOTE! Doesnt work - time remaining is not being calculated correctly. If enabled, this must be fixed.

        # SLC1: Time Remaining Fraction
        phi_time_remaining = float(time_remaining) / shift_length_safe

        # SLC2: Functional Bikes Time Penalty — urgency to flush bikes as shift ends
        phi_time_penalty = veh_load * (1.0 - phi_time_remaining)

        # SLC3: Reachable Imbalance Fraction — fraction of imbalance at reachable stations
        phi_reachable = (
            float(np.sum(np.abs(func - target) * (dist_to_stations <= time_remaining))
                  / total_imbalance)
            if total_imbalance >= 1e-6 else 0.0
        )

        cat_c = [phi_time_remaining, phi_time_penalty, phi_reachable]

        # SLC4: Recoverable Imbalance Fraction — reachable imbalance vs serviceable capacity
        reachable_imbalance = float(np.sum(np.abs(func - target) * (dist_to_stations <= time_remaining)))
        serviceable_capacity = max(0.0, K_safe - float(depot_cargo_veh))
        phi_recoverable_imbalance = min(reachable_imbalance, serviceable_capacity) / max(total_imbalance, 1.0)
        cat_c.append(phi_recoverable_imbalance)

        # SLC5: Imbalance Hotspot Distance — travel distance to worst-imbalance stations
        max_dist = max(float(np.max(dist_to_stations)), 1.0)
        imbalance = np.abs(func - target)
        if np.any(imbalance > 0.0):
            hotspot_count = min(5, N)
            hotspot_idx = np.argpartition(imbalance, -hotspot_count)[-hotspot_count:]
            hotspot_weights = imbalance[hotspot_idx]
            phi_imbalance_hotspot_dist = float(np.average(dist_to_stations[hotspot_idx], weights=hotspot_weights)) / max_dist
        else:
            phi_imbalance_hotspot_dist = 0.0
        cat_c.append(phi_imbalance_hotspot_dist)

        features.extend(cat_c)

    # =========================================================================
    # Pillar 5: Destination-Local Features
    # =========================================================================
    if not destination_features_enabled:
        return np.array(features, dtype=np.float32)

    if next_station_idx >= 0 and next_station_idx < N:
        nxt_func   = float(func[next_station_idx])
        nxt_onsite = float(onsite[next_station_idx])
        nxt_depot  = float(depot[next_station_idx])
        nxt_tgt    = float(target_safe[next_station_idx])
        nxt_cap    = float(max(capacities[next_station_idx], 1.0))
        nxt_cap_rem = float(max(capacities[next_station_idx] - target_safe[next_station_idx], 1.0))
        phi_dest_starv  = max(0.0, (nxt_tgt - nxt_func) / nxt_tgt)
        phi_dest_cong   = max(0.0, (nxt_func - nxt_tgt) / nxt_cap_rem)
        phi_dest_onsite = nxt_onsite / nxt_cap
        phi_dest_depot  = nxt_depot / nxt_cap
    else:
        phi_dest_starv = phi_dest_cong = phi_dest_onsite = phi_dest_depot = 0.0

    phi_dest_travel = dist_to_next / max(max_travel_time, 1.0)

    phi_roi_starv = (phi_dest_starv / max(phi_dest_travel, 0.05)) / 20.0

    # Current-station local features (post-decision).
    # These change 0.1–0.5 between candidates (vs ~0.004 for global features),
    # giving the VFA 50x better discrimination for maintenance and rebalancing decisions.
    if cur_station_idx >= 0 and cur_station_idx < N:
        cur_cap    = float(max(capacities[cur_station_idx], 1.0))
        cur_onsite = float(onsite[cur_station_idx])
        cur_func   = float(func[cur_station_idx])
        cur_tgt    = float(target_safe[cur_station_idx])
        phi_cur_onsite_frac  = cur_onsite / cur_cap
        phi_cur_func_deficit = max(0.0, cur_tgt - cur_func) / cur_cap
    else:
        phi_cur_onsite_frac  = 0.0
        phi_cur_func_deficit = 0.0

    # End-of-shift interactions. Raw time alone is not actionable, so expose it
    # through cargo and route terms instead.
    _service_window_min = max((SERVICE_TIME_TO - SERVICE_TIME_FROM) * 60.0, 1.0)
    _clock_min = current_time_minutes % 1440.0
    _close_min = SERVICE_TIME_TO * 60.0
    phi_shift_remaining = max(0.0, _close_min - _clock_min) / _service_window_min
    phi_late = 1.0 - phi_shift_remaining
    phi_func_late_pressure = veh_load * phi_late
    phi_depot_late_pressure = (depot_cargo_veh / K_safe) * phi_late
    phi_total_cargo_frac = (func_cargo_veh + depot_cargo_veh) / K_safe
    phi_non_depot_late_load_pressure = (0.0 if next_is_depot else phi_total_cargo_frac) * phi_late
    phi_late_depot_return = (1.0 if next_is_depot else 0.0) * phi_late

    # End-of-shift depot feasibility features.
    # dist_to_depot is travel time (minutes) from the candidate destination to the depot.
    # These let the VFA learn to avoid actions that strand cargo overnight.
    time_remaining_abs = max(0.0, _close_min - _clock_min)
    _depot_slack = time_remaining_abs - dist_to_depot
    phi_depot_slack = max(-1.0, min(1.0, _depot_slack / _service_window_min))
    phi_can_return = 1.0 if time_remaining_abs > dist_to_depot else 0.0
    phi_depot_urgency = min(1.0, dist_to_depot / max(time_remaining_abs, 1.0))

    features.extend([
        phi_dest_starv, phi_dest_cong, phi_dest_onsite, phi_dest_depot,
        phi_dest_travel, phi_roi_starv,
        phi_cur_onsite_frac, phi_cur_func_deficit,
        phi_func_late_pressure, phi_depot_late_pressure,
        phi_non_depot_late_load_pressure, phi_late_depot_return,
        phi_depot_slack, phi_can_return, phi_depot_urgency,
    ])

    return np.array(features, dtype=np.float32)



def as_dict(
    phi: np.ndarray,
    maintenance_enabled: bool = True,
    logistics_enabled: bool = False,
    demand_horizon_enabled: bool = False,
    destination_features_enabled: bool = True,
) -> dict:
    """Return a labelled dict of a computed feature vector."""
    names = get_feature_names(maintenance_enabled, logistics_enabled, demand_horizon_enabled, destination_features_enabled)
    assert len(phi) == len(names), (
        f"phi has {len(phi)} elements but {len(names)} names are registered. "
        f"Check maintenance_enabled={maintenance_enabled}, "
        f"logistics_enabled={logistics_enabled}, "
        f"demand_horizon_enabled={demand_horizon_enabled}."
    )
    return dict(zip(names, phi.tolist()))
