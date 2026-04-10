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
  I_i^func   functional bikes at station i  (post-decision)
  Î_i^func   time-indexed target inventory at station i
  C_i        physical rack capacity of station i
  λ_i        anticipated net demand at station i (+ out, - in)
  q_v^func   functional bikes on the vehicle (post-decision)
  K          vehicle capacity
  dist(v,i)  travel time from vehicle location to station i (minutes)
  t_rem      time remaining in shift (minutes)
  L          total shift length (minutes)

Features (Category A - Base Rebalancing)
────────
  φ_1  rebalancing_imbalance         Σ_i |I_i^func - Î_i^func| / (0.5 * Σ_i C_i)
  φ_2  anticipated_demand_shortfall  Σ_i [starv_risk_i + cong_risk_i] / (Λ_max / √N)
  φ_4  squared_starvation_penalty    (1/N) * Σ_i (max(0, Î_i - I_i) / Î_i)^2
  φ_5  squared_congestion_penalty    (1/N) * Σ_i (max(0, I_i - Î_i) / (C_i - Î_i))^2
  φ_6  proximity_to_demand_gravity   1 - (1/Φ_max) * Σ_i (|λ_i| / (dist(v,i)+1))
  φ_6a starvation_gravity            (q_v^func / K) * Σ_i (outflow_i * 1[I_i<Î_i] / (dist(v,i)+1)) / Φ_max
  φ_6b congestion_gravity            (free_v / K) * Σ_i (inflow_i * 1[I_i>Î_i] / (dist(v,i)+1)) / Φ_max
  φ_r1 imbalance_weighted_distance   Σ_i |I_i - Î_i| * dist(v,i) / (0.5*Σ C_i * max_dist)
  φ_r2 starvation_severity_max       max_i (max(0, Î_i - I_i) / Î_i)
  φ_r3 congestion_severity_max       max_i (max(0, I_i - Î_i) / (C_i - Î_i))
  φ_r4 station_starvation_count      Σ_i 1[I_i < Î_i] / N
  φ_w  work_ratio                    Σ_i |I_i - Î_i| / (4 * K)  — van trips needed
  φ_c  imbalance_concentration       max_i |I_i - Î_i| / Σ_i |I_i - Î_i|  — tractability

Features (Category B - Maintenance, appended if enabled)
────────
  φ_7  trailer_cannibalization       q_v^depot / K
  φ_8  onsite_backlog                Σ_i I_i^onsite / F
  φ_9  demand_weighted_depot         Σ_i (I_i^depot x |λ_i|) / Λ_max
  φ_10 depot_pull                    φ_7 x dist(v, depot) / 30
  φ_m  maintenance_urgency           φ_8 x φ_r4  — broken bikes AND many starving stations

Features (Category C - End-of-Day Anticipatory, appended if enabled)
────────
  φ_11 time_remaining_fraction       t_rem / L  ∈ [0, 1]
  φ_12 functional_bikes_time_penalty (q_v^func / K) x (1 - φ_11)
  φ_rch reachable_imbalance_fraction Σ_i |I_i-Î_i|*1[dist≤t_rem] / Σ_i |I_i-Î_i|
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from typing import List


# ── Feature registry ───────────────────────────────────────────────────────────────
def get_feature_names(
    maintenance_enabled: bool = True,
    shift_timing_enabled: bool = False,
    temporal_enabled: bool = False,
) -> list:
    """Returns the canonical list of feature names based on active modules."""

    # Category A: Base Rebalancing (Always active)
    names = [
        "rebalancing_imbalance",
        "anticipated_demand_shortfall",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "proximity_to_demand_gravity",
        "starvation_gravity",            # Van load × proximity to starving stations
        "congestion_gravity",            # Van free space × proximity to congested stations
        "imbalance_weighted_distance",   # Recoverability — how far away is the imbalance?
        "starvation_severity_max",       # Worst-case station starvation ratio
        "congestion_severity_max",       # Worst-case station congestion ratio (symmetric)
        "station_starvation_count",      # Fraction of stations currently below target
        "work_ratio",                    # NEW: Total imbalance / (4 * K) — trips needed
        "imbalance_concentration",       # NEW: max station imbalance / total — tractability
    ]

    # Category B: Maintenance Features
    if maintenance_enabled:
        names.extend([
            "trailer_cannibalization",
            "global_onsite_backlog",
            "demand_weighted_depot_backlog",
            "depot_pull",
            "maintenance_urgency",       # NEW: onsite_backlog × starvation_count
        ])

    # Category C: Shift Timing Features
    if shift_timing_enabled:
        names.extend([
            "time_remaining_fraction",
            "functional_bikes_time_penalty",
            "reachable_imbalance_fraction",  # NEW: fraction of imbalance reachable before EOD
        ])

    # Category D: Temporal Demand Features
    if temporal_enabled:
        names.extend([
            "time_of_day_fraction",          # Where in the 24h cycle are we?
            "day_of_week_fraction",          # Where in the weekly cycle are we?
            "hours_until_peak_fraction",     # How far until next morning rush?
            "multi_horizon_starvation_risk", # Integrated starvation over next H hours
            "temporal_demand_gradient",      # Is demand rising or falling next hour?
        ])

    return names


_MORNING_PEAK_HOUR: int = 8  # 08:00 — bike-sharing morning rush anchor


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
    maintenance_enabled: bool = True,
    shift_timing_enabled: bool = False,
    time_remaining: float = None,
    shift_length: float = 1440.0,
    temporal_enabled: bool = False,
    current_time_minutes: float = 0.0,
    current_day_of_week: int = 0,
    target_matrix: np.ndarray = None,   # shape (7, 24, N) — for multi-horizon look-ahead
    horizon_hours: int = 4,             # how many hours ahead to integrate
) -> np.ndarray:

    features = []

    # ── Safe Denominators ──
    vehicle_capacity_safe = max(float(vehicle_capacity), 1.0)
    lambda_max_safe = max(lambda_max_system, 1.0)
    max_gravity_safe = max(max_gravity, 1.0)
    K = max(vehicle_capacity, 1)
    N = len(func)
    F = max(fleet_size, 1.0)

    # ── Shared pre-computations (reused across multiple features) ──
    total_cap_half  = 0.5 * np.sum(capacities)
    total_imbalance = np.sum(np.abs(func - target))   # reused by φ_1, φ_w, φ_c, φ_rch

    # =========================================================================
    # Category A: Base Rebalancing Features
    # =========================================================================

    # φ_1: Rebalancing Imbalance
    phi_1 = total_imbalance / max(total_cap_half, 1.0)

    expected_outflow = np.maximum(0, -activity)  # negative activity = net rentals
    expected_inflow  = np.maximum(0,  activity)  # positive activity = net returns

    starvation_risk = np.maximum(0, expected_outflow - func)
    congestion_risk = np.maximum(0, func + expected_inflow - capacities)

    # φ_2: Anticipated Demand Shortfall
    # Normalised by lambda_max / sqrt(N): network-level shortfall grows with sqrt(N)
    # independent station contributions, not N.  φ_2 ∈ [0, 1].
    phi_2 = np.clip(
        np.sum(starvation_risk + congestion_risk) / (lambda_max_safe / np.sqrt(N)),
        0.0, 1.0,
    )

    # φ_3: Vehicle Functional Load (intermediate — used in interaction terms below, not a feature itself)
    phi_3 = func_cargo_veh / vehicle_capacity_safe

    # ── Debug: Target state audit (fires ~1% of calls to prevent terminal flood) ──
    if np.random.rand() < 0.01:
        network_starv_actual = np.sum(np.maximum(0, target - func))
        zero_targets = np.sum(target <= 0.0)
        if network_starv_actual == 0.0 and np.sum(func) < 100:
            print(f"\n[DEBUG 2 - TARGETS] City is empty (func={np.sum(func)}), but VFA sees 0 starvation.")
            print(f"  -> Why? Because target sum is: {np.sum(target):.1f}")
            print(f"  -> Number of stations with Target = 0: {zero_targets} out of {N}")
            if zero_targets > (N * 0.8):
                print(f"  -> ALARM: 80%+ of your network has a target of 0. Starvation penalty is mathematically impossible.")

    # φ_4: Squared Starvation Penalty
    target_safe = np.maximum(1.0, target)
    starv_ratio = np.maximum(0, target - func) / target_safe
    phi_4 = np.sum(starv_ratio**2) / N

    # ── Diagnostic: guaranteed one-shot printout if phi_4 == 0 ──
    if phi_4 == 0.0 and not getattr(extract, "_has_printed_phi4", False):
        print(f"\n[DIAGNOSTIC] phi_4 evaluated to exactly 0.0!")
        print(f"  -> Max target in the city:   {np.max(target)}")
        print(f"  -> Total func in the city:   {np.sum(func)}")
        zero_targets = np.sum(target <= 0.0)
        print(f"  -> Stations with Target = 0: {zero_targets} out of {N}")
        if zero_targets > (N * 0.8):
            print(f"  -> CONCLUSION: Culprit 1. Historical target algorithm is outputting 0 for almost everything.")
        else:
            print(f"  -> CONCLUSION: Culprit 2 or 3. Targets are healthy — van solved starvation, or city is over-saturated.")
        extract._has_printed_phi4 = True

    # φ_5: Squared Congestion Penalty
    cap_rem    = np.maximum(1.0, capacities - target)
    cong_ratio = np.maximum(0, func - target) / cap_rem
    phi_5 = np.sum(cong_ratio**2) / N

    # φ_6: LACK OF Proximity to Demand Gravity (inverted: penalty for being far from demand)
    raw_phi_6 = np.sum(np.abs(activity) / (dist_to_stations + 1.0)) / max_gravity_safe
    phi_6 = 1.0 - np.clip(raw_phi_6, 0.0, 1.0)

    # ── Interaction terms: van state × spatial demand structure ──────────────

    # Intermediate: van free-space fraction (used in φ_6b)
    free_space  = vehicle_capacity_safe - func_cargo_veh - depot_cargo_veh
    phi_3b_base = max(0.0, free_space) / vehicle_capacity_safe

    # φ_6a: Starvation Gravity — penalty for holding bikes near starving stations
    outflow             = np.maximum(0, -activity)
    starving_mask       = func < target
    starvation_grav_sum = np.sum((outflow * starving_mask) / (dist_to_stations + 1.0))
    phi_6a = np.clip(phi_3 * (starvation_grav_sum / max_gravity_safe), 0.0, 1.0)

    # φ_6b: Congestion Gravity — penalty for empty van near congested stations
    inflow              = np.maximum(0, activity)
    congested_mask      = func > target
    congestion_grav_sum = np.sum((inflow * congested_mask) / (dist_to_stations + 1.0))
    phi_6b = np.clip(phi_3b_base * (congestion_grav_sum / max_gravity_safe), 0.0, 1.0)

    # ── Debug: Gravity masking check (fires ~1% of calls) ──
    if np.random.rand() < 0.01:
        if starvation_grav_sum > 0.5 and phi_6a == 0.0:
            print(f"\n[DEBUG 3 - GRAVITY MASKING] Massive starvation gravity ({starvation_grav_sum:.2f}), but phi_6a = 0.")
            print(f"  -> phi_3 = {phi_3:.2f}  (Func Cargo: {func_cargo_veh})")
            if phi_3 == 0.0:
                print(f"  -> CONFIRMED: Feature is masked because the van is empty.")

    # φ_r1: Imbalance Weighted Distance — recoverability (distant imbalance = deferred cost)
    max_dist_safe = max(float(np.max(dist_to_stations)), 1.0)
    phi_r1 = np.clip(
        np.sum(np.abs(func - target) * dist_to_stations) / (total_cap_half * max_dist_safe),
        0.0, 1.0,
    )

    # φ_r2: Worst-case Station Starvation Ratio (reuses starv_ratio from φ_4)
    # 1.0 = at least one station has zero bikes against a non-zero target.
    phi_r2 = float(np.max(starv_ratio))

    # φ_r3: Worst-case Station Congestion Ratio (symmetric to φ_r2)
    cong_ratio_per_station = np.maximum(0, func - target) / np.maximum(1.0, capacities - target)
    phi_r3 = float(np.max(cong_ratio_per_station))

    # φ_r4: Fraction of Stations Below Target (breadth of starvation vs. φ_4 depth)
    phi_r4 = float(np.sum(func < target)) / N

    # φ_w: Work Ratio — total van trips needed to fix all imbalance
    # Normalised so that 4 full one-way van loads → 1.0 (saturated / barely recoverable).
    # Key long-term signal for rollout tail value: can the network be recovered at all?
    phi_work = np.clip(total_imbalance / max(4.0 * vehicle_capacity_safe, 1.0), 0.0, 1.0)

    # φ_c: Imbalance Concentration — tractability of the remaining work
    # max_station_imbalance / total_imbalance.
    # High (→1): one station dominates; fixable with a single well-placed visit.
    # Low (→1/N): imbalance is spread uniformly; many trips needed regardless of van state.
    if total_imbalance < 1e-6:
        phi_conc = 0.0
    else:
        phi_conc = float(np.max(np.abs(func - target))) / total_imbalance

    # Append Category A (clipped for numeric stability)
    cat_a = np.clip(
        [phi_1, phi_2, phi_4, phi_5, phi_6, phi_6a, phi_6b,
         phi_r1, phi_r2, phi_r3, phi_r4, phi_work, phi_conc],
        0.0, 1.0,
    ).tolist()
    features.extend(cat_a)

    # =========================================================================
    # Category B: Maintenance Features
    # =========================================================================
    if maintenance_enabled:
        # φ_7: Trailer Cannibalization (q_v^depot / K)
        phi_7 = float(depot_cargo_veh) / K

        # φ_8: Global Onsite Backlog (Σ_i I_i^onsite / F)
        phi_8 = float(np.sum(onsite)) / F

        # φ_9: Demand-Weighted Depot Backlog — broken bikes at high-activity stations
        phi_9 = float(np.sum(depot * np.abs(activity))) / lambda_max_safe

        # φ_10: Depot Pull — urgency to return grows as trailer fills and depot distance shrinks
        phi_10 = phi_7 * (dist_to_depot / 30.0)

        # φ_m: Maintenance Urgency — compound signal: onsite backlog × starvation breadth
        # High when many stations are BOTH below target AND have onsite broken bikes.
        # Captures the compounding failure mode where maintenance blocks rebalancing recovery.
        phi_maint = phi_8 * phi_r4

        cat_b = np.clip([phi_7, phi_8, phi_9, phi_10, phi_maint], 0.0, 1.0).tolist()
        features.extend(cat_b)

    # =========================================================================
    # Category C: End-of-Day Timing Features
    # =========================================================================
    if shift_timing_enabled:
        if time_remaining is None:
            time_remaining = shift_length

        shift_length_safe = max(shift_length, 1.0)

        # φ_11: Time Remaining Fraction ∈ [0, 1]
        phi_11 = np.clip(float(time_remaining) / shift_length_safe, 0.0, 1.0)

        # φ_12: Functional Bikes Time Penalty — urgency to flush functional bikes as shift ends
        phi_12 = phi_3 * (1.0 - phi_11)

        # φ_rch: Reachable Imbalance Fraction
        # What fraction of total imbalance can the vehicle physically reach before the shift ends?
        # A station is reachable if dist(v, i) ≤ time_remaining.
        # Low value late in shift → tail cost is largely unrecoverable, so long-term value is low.
        if total_imbalance < 1e-6:
            phi_reachable = 0.0
        else:
            reachable_mask = dist_to_stations <= time_remaining
            phi_reachable  = float(
                np.sum(np.abs(func - target) * reachable_mask) / total_imbalance
            )

        cat_c = np.clip([phi_11, phi_12, phi_reachable], 0.0, 1.0).tolist()
        features.extend(cat_c)

    # =========================================================================
    # Category D: Temporal Demand Features
    # =========================================================================
    if temporal_enabled:
        current_hour = int((current_time_minutes // 60) % 24)
        hours_until_morning_peak = (_MORNING_PEAK_HOUR - current_hour) % 24

        # φ_T1: Where in the 24h cycle? (0 = midnight, 0.5 = noon)
        phi_t1 = current_hour / 24.0

        # φ_T2: Where in the weekly cycle? (0 = Monday, 6/7 ≈ Sunday evening)
        phi_t2 = current_day_of_week / 7.0

        # φ_T3: How close is the next morning rush? (0 = at peak, 1 = 12+ hours away)
        phi_t3 = np.clip(hours_until_morning_peak / 12.0, 0.0, 1.0)

        # φ_D1: Multi-horizon starvation risk — integrate expected starvation over next H hours
        if target_matrix is not None:
            abs_hour_now      = int(current_time_minutes // 60)
            total_future_risk = 0.0
            for h in range(1, horizon_hours + 1):
                abs_hour_future = abs_hour_now + h
                future_hour     = abs_hour_future % 24
                future_day      = (current_day_of_week + abs_hour_future // 24) % 7
                future_target   = target_matrix[future_day, future_hour]
                total_future_risk += np.sum(np.maximum(0.0, future_target - func))
            phi_d1 = np.clip(
                total_future_risk / (horizon_hours * max(total_cap_half, 1.0)),
                0.0, 1.0,
            )
        else:
            phi_d1 = 0.0

        # φ_D2: Temporal Demand Gradient — is the total network target rising or falling?
        # Positive = more bikes needed soon.  Range: [-1, 1].
        if target_matrix is not None:
            abs_hour_next   = int(current_time_minutes // 60) + 1
            next_hour       = abs_hour_next % 24
            next_day        = (current_day_of_week + abs_hour_next // 24) % 7
            next_target_sum = float(np.sum(target_matrix[next_day, next_hour]))
            phi_d2 = np.clip(
                (next_target_sum - float(np.sum(target))) / max(total_cap_half, 1.0),
                -1.0, 1.0,
            )
        else:
            phi_d2 = 0.0

        cat_d = [phi_t1, phi_t2, phi_t3, phi_d1, phi_d2]
        features.extend(cat_d)

    return np.array(features, dtype=np.float64)


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
