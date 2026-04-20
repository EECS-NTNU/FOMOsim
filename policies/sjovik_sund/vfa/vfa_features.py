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

──────────────────────────────────────────────────────────────────────────────
Notation
────────
  I_i       functional bikes at station i  (post-decision)
  T_i       time-indexed target inventory at station i
  C_i       physical rack capacity of station i
  λ_i       anticipated net demand at station i (+ out, - in)
  q_func    functional bikes on the vehicle (post-decision)
  q_depot   broken bikes on the vehicle (post-decision)
  K         vehicle capacity
  d_i       travel time from vehicle location to station i (minutes)
  t_rem     time remaining in shift (minutes)
  L         total shift length (minutes)
  F         total fleet size (bikes)
  N         number of stations

──────────────────────────────────────────────────────────────────────────────
Category A  —  Base Rebalancing  (always active)
──────────────────────────────────────────────────────────────────────────────
  A1  rebalancing_imbalance         Σ_i |I_i - T_i| / (0.5 * Σ_i C_i)
  A3  squared_starvation_penalty    (1/N) Σ_i (max(0, T_i-I_i) / T_i)^2
  A4  squared_congestion_penalty    (1/N) Σ_i (max(0, I_i-T_i) / (C_i-T_i))^2
  A5  exponential_starvation_penalty (1/N) Σ_i (exp(3 * starv_ratio_i) - 1) / (exp(3) - 1)
  A6  exponential_congestion_penalty (1/N) Σ_i (exp(3 * cong_ratio_i) - 1) / (exp(3) - 1)
  A9  starvation_severity_max       q95_i (max(0, T_i-I_i) / T_i)
  A10 congestion_severity_max       q95_i (max(0, I_i-T_i) / (C_i-T_i))
  A11 unmet_starvation_deficit      max(0, sum(T_i - I_i) - van_bikes) / (0.5*Σ C_i)
  A12 starvation_variance           variance of starvation ratios
  A14 imbalance_hotspot_distance    distance to the worst-imbalance hotspot, normalised
  A15 starvation_count              fraction of stations with severe starvation
  A16 congestion_count              fraction of stations with severe congestion
  A18 hotspot_imbalance_mass        share of imbalance concentrated in the worst hotspot stations

Category B  —  Maintenance  (appended if maintenance_enabled)
──────────────────────────────────────────────────────────────────────────────
  B1  trailer_cannibalization       q_depot / K
  B2  global_onsite_backlog         Σ_i onsite_i / F
  B3  demand_weighted_depot_backlog Σ_i (depot_i * |λ_i|) / Λ_max
  B4  depot_pull                    B1 * (dist_to_depot / 30)
  B5  maintenance_urgency           B2 * A11    ← broken bikes AND starving stations

Category C  —  End-of-Day Timing  (appended if shift_timing_enabled)
──────────────────────────────────────────────────────────────────────────────
  C1  time_remaining_fraction       t_rem / L
  C2  functional_bikes_time_penalty (q_func / K) * (1 - C1)
  C3  reachable_imbalance_fraction  Σ_i |I_i-T_i|*1[d_i≤t_rem] / Σ_i |I_i-T_i|
  C4  recoverable_imbalance_fraction fraction of imbalance reachable and serviceable now

Category D  —  Temporal Demand  (appended if temporal_enabled)
──────────────────────────────────────────────────────────────────────────────
  D4  projected_starvation_risk     Σ_i max(0, expected_rentals_i - I_i) / Λ_max
  D5  projected_congestion_risk     Σ_i max(0, expected_returns_i - free_docks_i) / Λ_max
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Feature registry
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_names(
    maintenance_enabled: bool = True,
    shift_timing_enabled: bool = False,
    temporal_enabled: bool = False,
) -> list:
    """Returns the list of feature names based on active modules."""

    # Category A: Base Rebalancing (always active)
    names = [
        "rebalancing_imbalance",          # A1
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "exponential_starvation_penalty", # A5
        "exponential_congestion_penalty", # A6
        "starvation_severity_max",        # A9
        "congestion_severity_max",        # A10
        "unmet_starvation_deficit",       # A11
        "starvation_variance",            # A12
        "imbalance_hotspot_distance",     # A14
        "starvation_count",               # A15
        "congestion_count",               # A16
        "hotspot_imbalance_mass",         # A18
    ]

    # Category B: Maintenance
    if maintenance_enabled:
        names.extend([
            "trailer_cannibalization",        # B1
            "global_onsite_backlog",          # B2
            "demand_weighted_depot_backlog",  # B3
            "depot_pull",                     # B4
            "maintenance_urgency",            # B5
        ])

    # Category C: End-of-Day Timing
    if shift_timing_enabled:
        names.extend([
            "time_remaining_fraction",        # C1
            "functional_bikes_time_penalty",  # C2
            "reachable_imbalance_fraction",   # C3
            "recoverable_imbalance_fraction", # C4
        ])

    # Category D: Temporal Demand
    if temporal_enabled:
        names.extend([
            # "sin_time_of_day",                # D1a
            # "cos_time_of_day",                # D1b
            # "sin_day_of_week",                # D2a
            # "cos_day_of_week",                # D2b
            # "hours_until_peak_fraction",      # D3
            "projected_starvation_risk",      # D4
            "projected_congestion_risk",      # D5
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
    activity: np.ndarray,
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
    shift_timing_enabled: bool = False,
    time_remaining: float = None,
    shift_length: float = 1440.0,
    temporal_enabled: bool = False,
    current_time_minutes: float = 0.0,
    current_day_of_week: int = 0,
    target_matrix: np.ndarray = None,  # shape (7, 24, N) — for multi-horizon look-ahead
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

    # ── Shared demand decomposition ───────────────────────────────────────────
    # If activity is 2D (horizon_hours, N), simulate inventory evolution for lost demand
    if activity.ndim == 2:
        horizon, N = activity.shape
        # Initialize arrays
        lost_rentals = np.zeros(N)
        lost_returns = np.zeros(N)
        inventory = func.copy().astype(float)
        onsite_broken = onsite.copy().astype(float)
        cap = capacities.copy().astype(float)
        for t in range(horizon):
            # Rentals (negative activity): demand to remove bikes
            rentals = np.maximum(0.0, -activity[t])
            # Returns (positive activity): demand to add bikes
            returns = np.maximum(0.0, activity[t])
            # Lost rentals: demand above available bikes
            lost_now = np.maximum(0.0, rentals - inventory)
            lost_rentals += lost_now
            # Fulfilled rentals: can't take more than available
            fulfilled_rentals = np.minimum(rentals, inventory)
            inventory -= fulfilled_rentals
            # Returns: add bikes, but can't exceed capacity minus broken bikes
            free_docks = np.maximum(0.0, cap - inventory - onsite_broken)
            lost_now_ret = np.maximum(0.0, returns - free_docks)
            lost_returns += lost_now_ret
            # Fulfilled returns: can't return more than free docks
            fulfilled_returns = np.minimum(returns, free_docks)
            inventory += fulfilled_returns
        outflow = np.sum(np.maximum(0.0, -activity), axis=0)
        inflow = np.sum(np.maximum(0.0, activity), axis=0)
    else:
        # Backward compatibility: treat as before
        outflow = np.maximum(0.0, -activity)
        inflow  = np.maximum(0.0,  activity)

    # ── Shared per-station starvation/congestion ratios (reused in A3, A9) ───
    target_safe  = np.maximum(1.0, target)
    cap_rem_safe = np.maximum(1.0, capacities - target)
    starv_ratio  = np.maximum(0.0, target - func) / target_safe   # (T_i - I_i) / T_i
    cong_ratio   = np.maximum(0.0, func - target) / cap_rem_safe  # (I_i - T_i) / (C_i - T_i)

    # =========================================================================
    # Category A: Base Rebalancing Features
    # =========================================================================

    # A1: Rebalancing Imbalance — total L1 deviation from target, normalised by half capacity
    phi_imbalance = total_imbalance / max(total_cap_half, 1.0)

    # A3: Squared Starvation Penalty — mean squared starvation depth
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

    # A4: Squared Congestion Penalty — mean squared congestion depth
    phi_congestion_sq = np.sum(cong_ratio**2) / N

    # A5: Exponential Starvation Penalty — empirical mapping to an exponential penalty to increase tail response
    phi_starvation_exp = np.sum((np.exp(3.0 * starv_ratio) - 1.0) / (np.exp(3.0) - 1.0)) / N

    # A6: Exponential Congestion Penalty
    phi_congestion_exp = np.sum((np.exp(3.0 * cong_ratio) - 1.0) / (np.exp(3.0) - 1.0)) / N

    # A9: Starvation Severity (Q95) — severity at the 95th percentile of stations
    # Robust to single persistent outliers while still tracking tail risk.
    phi_starvation_max = float(np.quantile(starv_ratio, 0.95))

    # A10: Congestion Severity (Q95) — symmetric 95th percentile congestion depth.
    phi_congestion_max = float(np.quantile(cong_ratio, 0.95))

    # A11: Unmet Starvation Deficit — is the van empty when the city is starving?
    total_starvation = np.sum(np.maximum(0.0, target - func))
    phi_unmet_starv = max(0.0, float(total_starvation - func_cargo_veh)) / max(total_cap_half, 1.0)

    # A12: Starvation Variance — spread of the starvation problem
    phi_starv_var = float(np.var(starv_ratio))

    # A14: Distance to the worst-imbalance hotspot.
    # Uses the stations with the largest absolute deviation from target,
    # so tiny above/below-target noise across the network does not dominate.
    max_dist = max(float(np.max(dist_to_stations)), 1.0)
    imbalance = np.abs(func - target)
    if np.any(imbalance > 0.0):
        hotspot_count = min(5, N)
        hotspot_idx = np.argpartition(imbalance, -hotspot_count)[-hotspot_count:]
        hotspot_weights = imbalance[hotspot_idx]
        phi_imbalance_hotspot_dist = float(np.average(dist_to_stations[hotspot_idx], weights=hotspot_weights)) / max_dist
    else:
        phi_imbalance_hotspot_dist = 0.0

    # A18: Hotspot Imbalance Mass — how concentrated is the imbalance mass
    # in the stations that matter most right now?
    if total_imbalance > 0.0:
        phi_hotspot_imbalance_mass = float(np.sum(imbalance[hotspot_idx])) / max(float(total_imbalance), 1.0) if np.any(imbalance > 0.0) else 0.0
    else:
        phi_hotspot_imbalance_mass = 0.0

    # A15: Severe Station Starvation Count
    # Fraction of stations missing >= 90% of their target.
    phi_starvation_cnt = float(np.sum(starv_ratio >= 0.9)) / N

    # A16: Severe Station Congestion Count
    # Fraction of stations exceeding target by >= 90% of remaining capacity.
    phi_congestion_cnt = float(np.sum(cong_ratio >= 0.9)) / N

    cat_a = [
        phi_imbalance, phi_starvation_sq, phi_congestion_sq, 
        phi_starvation_exp, phi_congestion_exp,
        phi_starvation_max, phi_congestion_max, phi_unmet_starv, phi_starv_var,
        phi_imbalance_hotspot_dist, phi_starvation_cnt, phi_congestion_cnt,
        phi_hotspot_imbalance_mass
    ]
    features.extend(cat_a)

    # =========================================================================
    # Category B: Maintenance Features
    # =========================================================================
    if maintenance_enabled:

        # B1: Trailer Cannibalization — fraction of van capacity used by broken bikes
        phi_cannibalization = float(depot_cargo_veh) / K

        # B2: Global Onsite Backlog — broken bikes waiting at stations, normalised by fleet
        phi_onsite_backlog = float(np.sum(onsite)) / F_safe

        # B3: Demand-Weighted Depot Backlog — broken bikes concentrated at high-activity stations
        phi_depot_backlog = float(np.sum(depot * np.abs(activity))) / lam_max_safe

        # B4: Depot Pull — urgency to return grows as trailer fills and depot distance shrinks
        phi_depot_pull = phi_cannibalization * (dist_to_depot / 30.0)

        # B5: Maintenance Urgency — compound signal: onsite backlog × starvation breadth
        # High when many stations are BOTH below target AND have unrepaired bikes.
        phi_maint_urgency = phi_onsite_backlog * phi_starvation_cnt

        cat_b = [
            phi_cannibalization, phi_onsite_backlog, phi_depot_backlog,
            phi_depot_pull, phi_maint_urgency
        ]
        features.extend(cat_b)

    # =========================================================================
    # Category C: End-of-Day Timing Features
    # =========================================================================
    if shift_timing_enabled:
        if time_remaining is None:
            time_remaining = shift_length
        shift_length_safe = max(shift_length, 1.0)

        #NOTE! Doesnt work - time remaining is not being calculated correctly. If enabled, this must be fixed.

        # C1: Time Remaining Fraction
        phi_time_remaining = float(time_remaining) / shift_length_safe

        # C2: Functional Bikes Time Penalty — urgency to flush bikes as shift ends
        phi_time_penalty = veh_load * (1.0 - phi_time_remaining)

        # C3: Reachable Imbalance Fraction
        # Fraction of total imbalance at stations reachable before shift ends (dist ≤ t_rem).
        # Low value late in the shift signals that the tail cost is largely unrecoverable.
        phi_reachable = (
            float(np.sum(np.abs(func - target) * (dist_to_stations <= time_remaining))
                  / total_imbalance)
            if total_imbalance >= 1e-6 else 0.0
        )

        cat_c = [phi_time_remaining, phi_time_penalty, phi_reachable]

        # C4: Recoverable Imbalance Fraction — how much current imbalance can
        # still be reached and serviced by the vehicle's current remaining capacity.
        reachable_imbalance = float(np.sum(np.abs(func - target) * (dist_to_stations <= time_remaining)))
        serviceable_capacity = max(0.0, K_safe - float(depot_cargo_veh))
        phi_recoverable_imbalance = min(reachable_imbalance, serviceable_capacity) / max(total_imbalance, 1.0)

        cat_c.append(phi_recoverable_imbalance)
        features.extend(cat_c)

    # =========================================================================
    # Category D: Temporal Demand Features
    # =========================================================================
    if temporal_enabled:
        current_hour              = int((current_time_minutes // 60) % 24)
        # D4: Projected Starvation Risk — true lost rentals over the horizon
        # D5: Projected Congestion Risk — true lost returns over the horizon
        if activity.ndim == 2:
            phi_projected_starv = float(np.sum(lost_rentals)) / lam_max_safe
            phi_projected_cong = float(np.sum(lost_returns)) / lam_max_safe
            print(f"\n[DEBUG] Lost rentals: {lost_rentals}, Lost returns: {lost_returns}")
        else:
            # Fallback: old logic
            print("\n[WARNING] Temporal features enabled but activity is not 2D. Using fallback logic which may be inaccurate.")
            phi_projected_starv = float(np.sum(np.maximum(0.0, outflow - func))) / lam_max_safe
            free_docks = np.maximum(0.0, capacities - func - onsite)
            phi_projected_cong = float(np.sum(np.maximum(0.0, inflow - free_docks))) / lam_max_safe
        cat_d = [phi_projected_starv, phi_projected_cong]
        features.extend(cat_d)

    return np.array(features, dtype=np.float32)

def as_dict(
    phi: np.ndarray,
    maintenance_enabled: bool = True,
    shift_timing_enabled: bool = False,
    temporal_enabled: bool = False,
) -> dict:
    """Return a labelled dict of a computed feature vector."""
    names = get_feature_names(maintenance_enabled, shift_timing_enabled, temporal_enabled)
    assert len(phi) == len(names), (
        f"phi has {len(phi)} elements but {len(names)} names are registered. "
        f"Check maintenance_enabled={maintenance_enabled}, "
        f"shift_timing_enabled={shift_timing_enabled}, "
        f"temporal_enabled={temporal_enabled}."
    )
    return dict(zip(names, phi.tolist()))
