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
  φ_2  anticipated_demand_shortfall  Σ_i [max(0, λ_i - I_i^func) + max(0, I_i^func - λ_i - C_i)] / Λ_max
  φ_3  vehicle_functional_load       q_v^func / K
  φ_4  squared_starvation_penalty    (1/N) * Σ_i (Δ_starv / Î_i^func)^2
  φ_5  squared_congestion_penalty    (1/N) * Σ_i (Δ_cong / (C_i - Î_i^func))^2
  φ_6  proximity_to_demand_gravity   (1/Φ_max) * Σ_i (|λ_i| / (dist(v, i) + 1))

Features (Category B - Maintenance, appended if enabled)
────────
  φ_7  trailer_cannibalization       q_v^depot / K
  φ_8  onsite_backlog                Σ_i I_i^onsite / N
  φ_9  demand_weighted_depot         Σ_i (I_i^depot x |λ_i|) / Λ_max
  φ_10 depot_pull                    φ_7 x dist(v, depot) / 30

Features (Category C - End-of-Day Anticipatory, appended if enabled)
────────
  φ_11 time_remaining_fraction       t_rem / L  ∈ [0, 1]
  φ_12 functional_bikes_time_penalty (q_v^func / K) x (1 - φ_11)
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
        #"vehicle_functional_load",       # (Old phi_3 - kept for baseline ablation tests)
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "proximity_to_demand_gravity",   # (Old phi_6 - kept for baseline ablation tests)
        "delivery_potential",            # NEW: Actionable Capacity (Starving)
        "pickup_potential",              # NEW: Actionable Capacity (Congested)
        "starvation_gravity",            # NEW: Actionable Gravity (Requires full van)
        "congestion_gravity",            # NEW: Actionable Gravity (Requires empty van)
        "imbalance_weighted_distance",   # NEW: Recoverability — how far away is the imbalance?
        "starvation_severity_max",       # NEW: Worst-case station starvation ratio
        "congestion_severity_max",       # NEW: Worst-case station congestion ratio (symmetric)
        "station_starvation_count",      # NEW: Fraction of stations currently below target
    ]

    # Category B: Maintenance Features
    if maintenance_enabled:
        names.extend([
            "trailer_cannibalization",
            "global_onsite_backlog",
            "demand_weighted_depot_backlog",
            "depot_pull",
        ])

    # Category C: Shift Timing Features
    if shift_timing_enabled:
        names.extend([
            "time_remaining_fraction",
            "functional_bikes_time_penalty",
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
    N = len(func) # stations)
    F = max(fleet_size, 1.0)
    #print(f"func sum: {np.sum(func)}, onsite sum: {np.sum(onsite)}, depot sum: {np.sum(depot)}, func_cargo_veh: {func_cargo_veh}, depot_cargo_veh: {depot_cargo_veh}")
    #print(f"[DEBUG] Safe Denominators - Vehicle Cap: {vehicle_capacity_safe}, Lambda Max: {lambda_max_safe}, Max Gravity: {max_gravity_safe}, Number of Stations: {N}, Total Fleet: {F}")


    # =========================================================================
    # Category A: Base Rebalancing Features
    # =========================================================================
    
    # φ_1: Rebalancing Imbalance
    total_cap_half = 0.5 * np.sum(capacities)
    phi_1 = np.sum(np.abs(func - target)) / max(total_cap_half, 1.0)

    expected_outflow = np.maximum(0, -activity)  # Outflow is negative
    expected_inflow = np.maximum(0, activity)    # Inflow is positive

    starvation_risk = np.maximum(0, expected_outflow - func)
    congestion_risk = np.maximum(0, func + expected_inflow - capacities)

    # φ_2: Anticipated Demand Shortfall
    # Normalized by lambda_max / sqrt(N) rather than the full system total.
    # sqrt(N) accounts for aggregation over N stations: a network-level shortfall grows
    # with sqrt(N) independent station contributions, not N.
    # phi_2 ∈ [0, 1]: 0 = no demand at risk, 1 = system-wide crisis (shortfall ≈ lambda_max/sqrt(N)).
    phi_2 = np.clip(
        np.sum(starvation_risk + congestion_risk) / (lambda_max_safe / np.sqrt(N)),
        0.0, 1.0,
    )

    # φ_3: Vehicle Functional Load (Baseline)
    phi_3 = func_cargo_veh / vehicle_capacity_safe


    # --- INJECT DEBUG BLOCK 2: TARGET STATE AUDIT ---
    # Runs ~1% of the time to prevent terminal flood
    if np.random.rand() < 0.7: 
        network_starv_actual = np.sum(np.maximum(0, target - func))
        zero_targets = np.sum(target <= 0.0)
        
        if network_starv_actual == 0.0 and np.sum(func) < 100:
            print(f"\n[DEBUG 2 - TARGETS] City is empty (func={np.sum(func)}), but VFA sees 0 starvation.")
            print(f"  -> Why? Because target sum is: {np.sum(target):.1f}")
            print(f"  -> Number of stations with Target = 0: {zero_targets} out of {N}")
            if zero_targets > (N * 0.8):
                print(f"  -> ALARM: 80%+ of your network has a target of 0. Starvation penalty is mathematically impossible.")
    # ------------------------------------------------

    # φ_4: Squared Starvation Penalty
    target_safe = np.maximum(1.0, target)
    starv_ratio = np.maximum(0, target - func) / target_safe
    phi_4 = np.sum(starv_ratio**2) / N

    # --- INJECT GUARANTEED PHI_4 DIAGNOSTIC ---
    if phi_4 == 0.0 and not getattr(extract, "_has_printed_phi4", False):
        print(f"\n[DIAGNOSTIC] phi_4 evaluated to exactly 0.0!")
        print(f"  -> Max target in the city:   {np.max(target)}")
        print(f"  -> Total func in the city:   {np.sum(func)}")
        
        # Are the targets just zeroes?
        zero_targets = np.sum(target <= 0.0)
        print(f"  -> Stations with Target = 0: {zero_targets} out of {N}")
        
        if zero_targets > (N * 0.8):
            print(f"  -> CONCLUSION: Culprit 1. Your historical target algorithm is outputting 0 for almost everything.")
        else:
            print(f"  -> CONCLUSION: Culprit 2 or 3. The targets are healthy, meaning the van's post-decision state perfectly solved the starvation, or the city is over-saturated with bikes.")
        
        # Stop printing after the first catch
        extract._has_printed_phi4 = True
    # ------------------------------------------

    # φ_5: Squared Congestion Penalty
    cap_rem = np.maximum(1.0, capacities - target)
    cong_ratio = np.maximum(0, func - target) / cap_rem
    phi_5 = np.sum(cong_ratio**2) / N

    # φ_6: LACK OF Proximity to Demand Gravity (General Bonus -> Inverted to Penalty)
    raw_phi_6 = np.sum(np.abs(activity) / (dist_to_stations + 1.0)) / max_gravity_safe
    phi_6 = 1.0 - np.clip(raw_phi_6, 0.0, 1.0)
    
    ######### interaction terms ##########
    
    '''# φ_3A: Delivery Potential (Actionable Capacity)
    network_starvation = np.sum(np.maximum(0, target - func))
    phi_3a = phi_3 * (network_starvation / max(total_cap_half, 1.0))
    
    # φ_3B: Pickup Potential (Actionable Capacity)
    free_space = vehicle_capacity_safe - func_cargo_veh - depot_cargo_veh
    phi_3b_base = max(0.0, free_space) / vehicle_capacity_safe
    network_congestion = np.sum(np.maximum(0, func - target))
    phi_3b = phi_3b_base * (network_congestion / max(total_cap_half, 1.0))
    
    # φ_6A: Starvation Gravity (Actionable Gravity)
    outflow = np.maximum(0, -activity) # Negative activity means net rentals
    starving_mask = func < target
    starvation_grav_sum = np.sum((outflow * starving_mask) / (dist_to_stations + 1.0))
    phi_6a = phi_3 * (starvation_grav_sum / max_gravity_safe)
    
    # φ_6B: Congestion Gravity (Actionable Gravity)
    inflow = np.maximum(0, activity) # Positive activity means net returns
    congested_mask = func > target
    congestion_grav_sum = np.sum((inflow * congested_mask) / (dist_to_stations + 1.0))
    phi_6b = phi_3b_base * (congestion_grav_sum / max_gravity_safe)'''
    
    # φ_3A: Delivery Potential (Penalty: Holding bikes while network starves)
    network_starvation = np.sum(np.maximum(0, target - func))
    raw_phi_3a = phi_3 * (network_starvation / max(total_cap_half, 1.0))
    phi_3a = np.clip(raw_phi_3a, 0.0, 1.0)
    
    # φ_3B: Pickup Potential (Penalty: Empty van while network is congested)
    free_space = vehicle_capacity_safe - func_cargo_veh - depot_cargo_veh
    phi_3b_base = max(0.0, free_space) / vehicle_capacity_safe
    network_congestion = np.sum(np.maximum(0, func - target))
    raw_phi_3b = phi_3b_base * (network_congestion / max(total_cap_half, 1.0))
    phi_3b = np.clip(raw_phi_3b, 0.0, 1.0)
    
    # φ_6A: Starvation Gravity (Penalty: Full van parked near a starving station)
    outflow = np.maximum(0, -activity) # Negative activity means net rentals
    starving_mask = func < target
    starvation_grav_sum = np.sum((outflow * starving_mask) / (dist_to_stations + 1.0))
    raw_phi_6a = phi_3 * (starvation_grav_sum / max_gravity_safe)
    phi_6a = np.clip(raw_phi_6a, 0.0, 1.0)
    
    # φ_6B: Congestion Gravity (Penalty: Empty van parked near a congested station)
    inflow = np.maximum(0, activity) # Positive activity means net returns
    congested_mask = func > target
    congestion_grav_sum = np.sum((inflow * congested_mask) / (dist_to_stations + 1.0))
    raw_phi_6b = phi_3b_base * (congestion_grav_sum / max_gravity_safe)
    phi_6b = np.clip(raw_phi_6b, 0.0, 1.0)

    # --- INJECT DEBUG BLOCK 3: GRAVITY MASKING ---
    if np.random.rand() < 0.01:
        if starvation_grav_sum > 0.5 and phi_6a == 0.0:
            print(f"\n[DEBUG 3 - GRAVITY MASKING] Massive starvation gravity exists ({starvation_grav_sum:.2f}), but phi_6a evaluated to 0.0!")
            print(f"  -> Was the van empty? phi_3 = {phi_3:.2f} (Func Cargo: {func_cargo_veh})")
            if phi_3 == 0.0:
                print(f"  -> CONFIRMED: Actionable Gravity feature is hiding the network starvation because the van is empty.")
    # ---------------------------------------------
    
    ######################DEBUG: Print raw feature values before clipping (occasionally)######################
    # 1. Store the raw values before they get squashed
    raw_cat_a = [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_3a, phi_3b, phi_6a, phi_6b]

    # 2. --- DEBUG: Monitor Contextual Feature Clipping ---
    # We check if the contextual features are breaching the 1.0 ceiling.
    # Using a random threshold (e.g., 1%) prevents terminal spam since this 
    # function is called thousands of times per episode.
    if (phi_6a > 1.0 or phi_6b > 1.0) and np.random.rand() < 0.01:
        print(f"\n[DEBUG - CLIPPING] Contextual features exceeding 1.0 ceiling!")
        print(f"  -> phi_6a (Starv Gravity) Raw: {phi_6a:.3f}")
        print(f"  -> phi_6b (Cong Gravity)  Raw: {phi_6b:.3f}")
        print(f"  -> phi_3a (Del Potential) Raw: {phi_3a:.3f}")
        print(f"  -> phi_3b (Pick Potential)Raw: {phi_3b:.3f}")
        print(f"  * Note: These will be clipped to 1.0 for the VFA.\n")

    # 3. Apply the clip and extend as usual
    #cat_a = np.clip(raw_cat_a, 0.0, 1.0).tolist()
    #features.extend(cat_a)
    ############################################################################################################

    # φ_R1: Imbalance Weighted Distance
    # Captures recoverability: a station badly out of balance AND far away is harder to fix.
    # Normalized by total_cap_half * max possible travel time (capped at 60 min).
    max_dist_safe = max(float(np.max(dist_to_stations)), 1.0)
    phi_r1 = np.clip(
        np.sum(np.abs(func - target) * dist_to_stations) / (total_cap_half * max_dist_safe),
        0.0, 1.0,
    )

    # φ_R2: Worst-case Station Starvation Ratio
    # max_i (max(0, target_i - func_i) / target_i) — reuses starv_ratio already computed for φ_4.
    # 1.0 means at least one station has zero bikes against a non-zero target.
    phi_r2 = float(np.max(starv_ratio))

    # φ_R3: Worst-case Station Congestion Ratio (symmetric to φ_R2)
    # max_i (max(0, func_i - target_i) / (C_i - target_i))
    # 1.0 means at least one station is full against a non-full target.
    cong_ratio_per_station = np.maximum(0, func - target) / np.maximum(1.0, capacities - target)
    phi_r3 = float(np.max(cong_ratio_per_station))

    # φ_R4: Fraction of Stations Currently Below Target
    # Captures breadth of starvation (how many stations) vs. φ_4 which captures depth (how much).
    phi_r4 = float(np.sum(func < target)) / N

    # Append Category A (Clipped to ensure numeric stability for VFA)
    cat_a = np.clip([phi_1, phi_2, phi_4, phi_5, phi_6, phi_3a, phi_3b, phi_6a, phi_6b, phi_r1, phi_r2, phi_r3, phi_r4], 0.0, 1.0).tolist()
    features.extend(cat_a)

    # =========================================================================
    # Category B: Maintenance Features
    # =========================================================================
    if maintenance_enabled:
        # φ_7: Trailer Cannibalization (q_v^depot / K)
        phi_7 = float(depot_cargo_veh) / K

        # φ_8: Global Onsite Backlog (Σ_i I_i^onsite / N)
        # Scaled by global fleet or capacities in the future, for now N.
        phi_8 = float(np.sum(onsite)) / F 

        # φ_9: Demand-Weighted Depot Backlog
        # Penalize broken bikes at high-activity stations.
        phi_9 = float(np.sum(depot * np.abs(activity))) / lambda_max_safe

        # φ_10: Depot Pull
        # Naturally increases urgency to return to depot as trailer fills
        phi_10 = phi_7 * (dist_to_depot / 30.0)

        cat_b = np.clip([phi_7, phi_8, phi_9, phi_10], 0.0, 1.0).tolist()
        features.extend(cat_b)

    # =========================================================================
    # Category C: End-of-Day Timing Features
    # =========================================================================
    if shift_timing_enabled:
        if time_remaining is None:
            time_remaining = shift_length

        # φ_11: Time Remaining Fraction ∈ [0, 1]
        shift_length_safe = max(shift_length, 1.0)
        phi_11 = np.clip(float(time_remaining) / shift_length_safe, 0.0, 1.0)

        # φ_12: Functional Bikes Time Penalty
        # Creates urgency to flush functional bikes as shift ends.
        phi_12 = phi_3 * (1.0 - phi_11)

        cat_c = np.clip([phi_11, phi_12], 0.0, 1.0).tolist()
        features.extend(cat_c)

    # =========================================================================
    # Category D: Temporal Demand Features
    # =========================================================================
    if temporal_enabled:
        # Derive time coordinates from the simulation clock
        current_hour = int((current_time_minutes // 60) % 24)

        # Hours until next morning peak (e.g. 8am).
        # Result is in [0, 23]: 0 means we are AT peak, 23 means we just passed it.
        hours_until_morning_peak = (_MORNING_PEAK_HOUR - current_hour) % 24

        # φ_T1: Where in the 24h cycle are we? (0 = midnight, 0.5 = noon)
        phi_t1 = current_hour / 24.0

        # φ_T2: Where in the weekly cycle? (0 = Monday, 6/7 ≈ Sunday evening)
        phi_t2 = current_day_of_week / 7.0

        # φ_T3: How close is the next morning rush?
        # Normalized over a 12-hour window: 0 = peak is now, 1 = 12+ hours away
        phi_t3 = np.clip(hours_until_morning_peak / 12.0, 0.0, 1.0)

        # φ_D1: Multi-horizon starvation risk
        # Integrate expected starvation across the next `horizon_hours` target snapshots.
        # Uses the full 7×24×N target_matrix so the VFA can see upcoming demand pressure.
        if target_matrix is not None:
            abs_hour_now = int(current_time_minutes // 60)
            total_future_risk = 0.0
            for h in range(1, horizon_hours + 1):
                abs_hour_future = abs_hour_now + h
                future_hour = abs_hour_future % 24
                future_day  = (current_day_of_week + abs_hour_future // 24) % 7
                future_target = target_matrix[future_day, future_hour]
                total_future_risk += np.sum(np.maximum(0.0, future_target - func))
            phi_d1 = np.clip(
                total_future_risk / (horizon_hours * max(total_cap_half, 1.0)),
                0.0, 1.0,
            )
        else:
            phi_d1 = 0.0

        # φ_D2: Temporal Demand Gradient
        # Is the total network target rising or falling in the next hour?
        # Positive = more bikes will be needed soon (rising demand pressure).
        # Negative = demand pressure is easing next hour.
        # Range: [-1, 1]. Complements φ_D1 (cumulative risk) with a directional signal.
        if target_matrix is not None:
            abs_hour_next = int(current_time_minutes // 60) + 1
            next_hour = abs_hour_next % 24
            next_day = (current_day_of_week + abs_hour_next // 24) % 7
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