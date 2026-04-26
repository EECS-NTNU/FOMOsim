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
  CIM8  starvation_variance            variance of starvation ratios
  CIM9  starvation_count               fraction of stations with starvation ratio ≥ 0.9
  CIM10 congestion_count               fraction of stations with congestion ratio ≥ 0.9

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
  MP3  demand_weighted_depot_backlog   Σ_i (depot_i * |λ_i^net|) / Λ_max
  MP4  depot_pull                      MP1 * (dist_to_depot / 30)
  MP5  maintenance_urgency             MP2 * CIM9  ← broken bikes AND starving stations

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


# ─────────────────────────────────────────────────────────────────────────────
# Feature registry
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_names(
    maintenance_enabled: bool = True,
    logistics_enabled: bool = False,
    demand_horizon_enabled: bool = False,
) -> list:
    """Returns the canonical feature name list for the active operational pillars.

    Pillar 1 (Current System Imbalance) is always active.
    Pillar 2 (Future System Imbalance)   — demand_horizon_enabled
    Pillar 3 (Asset Health & Recovery)   — maintenance_enabled
    Pillar 4 (Spatial & Logistic)        — logistics_enabled
    """

    # Pillar 1: Current System Imbalance (CIM, always active)
    names = [
        "rebalancing_imbalance",          # CIM1
        "squared_starvation_penalty",     # CIM2
        "squared_congestion_penalty",     # CIM3
        "exponential_starvation_penalty", # CIM4
        "exponential_congestion_penalty", # CIM5
        "starvation_severity_max",        # CIM6
        "congestion_severity_max",        # CIM7
        "starvation_variance",            # CIM8
        "starvation_count",               # CIM9
        "congestion_count",               # CIM10
    ]

    # Pillar 2: Future System Imbalance (FIM)
    if demand_horizon_enabled:
        names.extend([
            "gross_starvation_risk",          # FIM1
            "gross_congestion_risk",          # FIM2
            "net_starvation_shortfall",       # FIM3
            "net_congestion_shortfall",       # FIM4
        ])

    # Pillar 3: Maintenance Pressure (MP)
    if maintenance_enabled:
        names.extend([
            "trailer_cannibalization",        # MP1
            "global_onsite_backlog",          # MP2
            "demand_weighted_depot_backlog",  # MP3
            "depot_pull",                     # MP4
            "maintenance_urgency",            # MP5
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

    return names


_MORNING_PEAK_HOUR: int = 8  # 08:00 — bike-sharing morning rush anchor


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
    total_stations: int,           # <-- NEW
    maintenance_enabled: bool = True,
    logistics_enabled: bool = False,
    time_remaining: Optional[float] = None,
    shift_length: float = 1440.0,
    demand_horizon_enabled: bool = True,
    current_time_minutes: float = 0.0,
    current_day_of_week: int = 0,
    target_matrix: Optional[np.ndarray] = None,  # shape (7, 24, N) — for multi-horizon look-ahead
    horizon_hours: int = 4,
) -> np.ndarray:

    features = []


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

    # CIM8: Starvation Variance — spread of the starvation problem
    phi_starv_var = float(np.var(starv_ratio))

    # CIM9: Severe Station Starvation Count — fraction with starvation ratio >= 0.9
    phi_starvation_cnt = float(np.sum(starv_ratio >= 0.9)) / N

    # CIM10: Severe Station Congestion Count — fraction with congestion ratio >= 0.9
    phi_congestion_cnt = float(np.sum(cong_ratio >= 0.9)) / N

    cat_a = [
        phi_imbalance, phi_starvation_sq, phi_congestion_sq,
        phi_starvation_exp, phi_congestion_exp,
        phi_starvation_max, phi_congestion_max, phi_starv_var,
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

        # FIM3: Net Starvation Shortfall — net outflow pressure with demand uncertainty
        net_std_dev = np.sqrt(gross_outflow + gross_inflow)
        phi_net_starv_shortfall = float(np.mean(np.maximum(0.0, net_activity + net_std_dev) / target_safe))

        # FIM4: Net Congestion Shortfall — net inflow pressure with demand uncertainty
        phi_net_cong_shortfall  = float(np.mean(np.maximum(0.0, -net_activity + net_std_dev) / cap_rem_safe))

        cat_d = [phi_gross_starv, phi_gross_cong, phi_net_starv_shortfall, phi_net_cong_shortfall]
        features.extend(cat_d)

    # =========================================================================
    # Pillar 3: Maintenance Pressure (MP)
    # =========================================================================
    if maintenance_enabled:

        # MP1: Trailer Cannibalization — fraction of van capacity used by broken bikes
        phi_cannibalization = float(depot_cargo_veh) / K

        # MP2: Global Onsite Backlog — broken bikes waiting at stations, normalised by fleet
        phi_onsite_backlog = float(np.sum(onsite)) / F_safe

        # MP3: Demand-Weighted Depot Backlog — broken bikes concentrated at high-activity stations
        phi_depot_backlog = float(np.sum(depot * np.abs(net_activity))) / lam_max_safe

        # MP4: Depot Pull — urgency to return grows as trailer fills and depot distance shrinks
        phi_depot_pull = phi_cannibalization * (dist_to_depot / 30.0)

        # MP5: Maintenance Urgency — compound signal: onsite backlog × starvation breadth
        # High when many stations are BOTH below target AND have unrepaired bikes.
        phi_maint_urgency = phi_onsite_backlog * phi_starvation_cnt

        cat_b = [
            phi_cannibalization, phi_onsite_backlog, phi_depot_backlog,
            phi_depot_pull, phi_maint_urgency
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

    return np.array(features, dtype=np.float32)

def as_dict(
    phi: np.ndarray,
    maintenance_enabled: bool = True,
    logistics_enabled: bool = False,
    demand_horizon_enabled: bool = False,
) -> dict:
    """Return a labelled dict of a computed feature vector."""
    names = get_feature_names(maintenance_enabled, logistics_enabled, demand_horizon_enabled)
    assert len(phi) == len(names), (
        f"phi has {len(phi)} elements but {len(names)} names are registered. "
        f"Check maintenance_enabled={maintenance_enabled}, "
        f"logistics_enabled={logistics_enabled}, "
        f"demand_horizon_enabled={demand_horizon_enabled}."
    )
    return dict(zip(names, phi.tolist()))
