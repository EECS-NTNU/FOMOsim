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
def get_feature_names(maintenance_enabled: bool = False, shift_timing_enabled: bool = False) -> List[str]:
    """
    Returns the appropriate feature names based on enabled feature categories.
    """
    features = [
        "rebalancing_imbalance",
        "anticipated_demand_shortfall",
        "vehicle_functional_load",
        "squared_starvation_penalty",
        "squared_congestion_penalty",
        "proximity_to_demand_gravity",
    ]
    
    if maintenance_enabled:
        features.extend([
            "trailer_cannibalization",
            "onsite_backlog",
            "demand_weighted_depot",
            "depot_pull",
        ])
        
    if shift_timing_enabled:
        features.extend([
            "time_remaining_fraction",
            "functional_bikes_time_penalty",
        ])
        
    return features


def extract(
    func:                  np.ndarray,   # (N,) functional bikes post-decision
    onsite:                np.ndarray,   # (N,) onsite-repairable bikes
    depot:                 np.ndarray,   # (N,) depot-level damaged bikes
    target:                np.ndarray,   # (N,) time-indexed target Î_i^func
    capacities:            np.ndarray,   # (N,) physical dock capacity C_i
    activity:              np.ndarray,   # (N,) anticipated net demand λ_i per station (+ rentals, - returns)
    dist_to_stations:      np.ndarray,   # (N,) travel time from vehicle to each station i
    func_cargo_veh:        float,        # q_v^func (post-decision)
    depot_cargo_veh:       float,        # q_v^depot  (post-decision)
    vehicle_capacity:      int,          # K
    dist_to_depot:         float,        # dist(v, depot) in simulation minutes
    lambda_max_system:     float,        # Λ_max (scaling constant for shortfall)
    max_gravity:           float,        # G_max (scaling constant for gravity)
    maintenance_enabled:   bool = False,
    shift_timing_enabled:  bool = False,
    time_remaining:        float = None, # t_rem – remaining shift time (minutes)
    shift_length:          float = 1440.0,  # reference shift length (L) in minutes
) -> np.ndarray:
    """
    Compute φ(S^x) from pre-resolved inventory arrays.
    Pure numpy - no simulator calls, no side-effects.
    """
    K = max(vehicle_capacity, 1)
    N = max(len(func), 1)   
    
    # Safely avoid divide-by-zero for network-wide constants
    lambda_max_safe = max(lambda_max_system, 1.0)
    max_gravity_safe = max(max_gravity, 1e-8)
    max_imbalance = max(1.0, 0.5 * np.sum(capacities))

    # =========================================================================
    # Category A: Base Rebalancing Features
    # =========================================================================
    
    # φ_1: Rebalancing Imbalance
    # Σ_i |I_i^func - Î_i^func| / (0.5 * Σ_i C_i)
    phi_1 = float(np.sum(np.abs(func - target))) / max_imbalance
    
    # φ_2: Anticipated Net Demand Shortfall
    # Σ_i [max(0, λ_i - I_i^func) + max(0, I_i^func - λ_i - C_i)] / Λ_max
    starv_risk = np.maximum(0, activity - func)
    cong_risk = np.maximum(0, (func - activity) - capacities)
    phi_2 = float(np.sum(starv_risk + cong_risk)) / lambda_max_safe

    # φ_3: Vehicle Functional Load
    # q_v^func / K
    phi_3 = float(func_cargo_veh) / K
    
    # φ_4: Squared Starvation Penalty
    # (1/N) * Σ_i (Δ_starv / max(1, Î_i^func))^2
    delta_starv = np.maximum(0, target - func)
    norm_starv = delta_starv / np.maximum(1.0, target)
    phi_4 = float(np.sum(norm_starv**2)) / N
    
    # φ_5: Squared Congestion Penalty
    # (1/N) * Σ_i (Δ_cong / max(1, C_i - Î_i^func))^2
    delta_cong = np.maximum(0, func - target)
    norm_cong = delta_cong / np.maximum(1.0, capacities - target)
    phi_5 = float(np.sum(norm_cong**2)) / N

    # φ_6: Proximity to Demand Gravity
    # (1/Φ_max) * Σ_i (|λ_i| / (dist(v, i) + 1))
    gravity_scores = np.abs(activity) / (dist_to_stations + 1.0)
    phi_6 = float(np.sum(gravity_scores)) / max_gravity_safe

    # Compile Category A (Clip all mathematically to [0, 1] just in case of float errors)
    features = np.clip([phi_1, phi_2, phi_3, phi_4, phi_5, phi_6], 0.0, 1.0).tolist()

    # =========================================================================
    # Category B: Maintenance Features
    # =========================================================================
    if maintenance_enabled:
        # φ_7: Trailer Cannibalization (q_v^depot / K)
        phi_7 = float(depot_cargo_veh) / K

        # φ_8: Global Onsite Backlog (Σ_i I_i^onsite / N)
        # Scaled by global fleet or capacities in the future, for now N.
        phi_8 = float(np.sum(onsite)) / N

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

    return np.array(features, dtype=np.float64)


def as_dict(phi: np.ndarray, maintenance_enabled: bool = True, shift_timing_enabled: bool = False) -> dict:
    """
    Return a labelled dict of a computed feature vector.
    """
    features = get_feature_names(maintenance_enabled, shift_timing_enabled)
    assert len(phi) == len(features), (
        f"phi has {len(phi)} elements but {len(features)} names are registered. "
        f"Check maintenance_enabled={maintenance_enabled}, shift_timing_enabled={shift_timing_enabled}."
    )
    return dict(zip(features, phi.tolist()))