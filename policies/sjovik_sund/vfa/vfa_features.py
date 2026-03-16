"""
vfa_features.py  –  Feature Vector φ(S^x) for the Time-Indexed Linear VFA

This is the CANONICAL definition of every feature used in the VFA.
To change the feature set:
  1. Edit get_feature_names().
  2. Edit the corresponding computation block in extract().
  3. Update the return array in extract() to match.
  4. Retrain – nothing else needs touching.

LinearVFAPolicy imports this module as the single source of truth via:
    - get_feature_names()
    - extract(...)
    - as_dict(...)

──────────────────────────────────────────────────────────────────────────────
Notation
────────
  I_i^func   functional bikes at station i  (post-decision)
  Î_i^func   time-indexed target inventory at station i
  I_i^onsite onsite-repairable bikes at station i
  I_i^depot  depot-level damaged bikes at station i
  λ_i        time-averaged arrival rate at station i
  q_v^func   functional bikes on the vehicle (post-decision)
  q_v^depot  depot-damaged bikes on the vehicle (post-decision)
  K          vehicle capacity
  dist(v,d)  travel time from vehicle location to nearest depot (minutes)

Features (Category A - Rebalancing)
────────
  φ_1  rebalancing_imbalance      Σ_i |I_i^func − Î_i^func| / N
  φ_2  vehicle_functional_load    q_v^func / K
  φ_3  demand_weighted_imbalance  Σ_i (|I_i^func − Î_i^func| × λ_i) / N

Features (Category B - Maintenance, appended if enabled)
────────
  φ_4  trailer_cannibalization    q_v^depot / K
  φ_5  onsite_backlog             Σ_i I_i^onsite / N
  φ_6  demand_weighted_depot      Σ_i (I_i^depot × λ_i) / N
  φ_7  depot_pull                 φ_4 × dist(v, depot) / 30
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from typing import List


# ── Feature registry ───────────────────────────────────────────────────────────────
def get_feature_names(maintenance_enabled: bool = True) -> List[str]:
    """
    Returns the appropriate feature names based on whether maintenance is enabled.
    Category A (Rebalancing):
      - rebalancing_imbalance
      - vehicle_functional_load
      - demand_weighted_imbalance
    Category B (Maintenance):
      - trailer_cannibalization
      - onsite_backlog
      - demand_weighted_depot
      - depot_pull
    """
    features = [
        "rebalancing_imbalance",
        "vehicle_functional_load",
        "demand_weighted_imbalance",
    ]
    if maintenance_enabled:
        features.extend([
            "trailer_cannibalization",
            "onsite_backlog",
            "demand_weighted_depot",
            "depot_pull",
        ])
    return features


def extract(
    func:             np.ndarray,   # (N,) functional bikes post-decision
    onsite:           np.ndarray,   # (N,) onsite-repairable bikes
    depot:            np.ndarray,   # (N,) depot-level damaged bikes
    target:           np.ndarray,   # (N,) time-indexed target Î_i^func
    activity:         np.ndarray,   # (N,) time-averaged arrival rate λ_i
    func_cargo_veh:   float,        # q_v^func (post-decision)
    depot_cargo_veh:  float,        # q_v^depot  (post-decision)
    vehicle_capacity: int,          # K
    dist_to_depot:    float,        # dist(v, depot) in simulation minutes
    maintenance_enabled: bool = True,
) -> np.ndarray:
    """
    Compute φ(S^x) from pre-resolved inventory arrays.

    Pure numpy – no simulator calls, no side-effects.
    All inputs are prepared by LinearVFAPolicy before calling this function.

    Args:
        func                : (N,) functional bike counts (post-decision)
        onsite              : (N,) onsite-repairable bike counts
        depot               : (N,) depot-level damaged bike counts
        target              : (N,) time-indexed target inventory Î_i^func
        activity            : (N,) time-averaged arrival rate λ_i per station
        func_cargo_veh      : vehicle functional-bike cargo after action (q_v^func)
        depot_cargo_veh     : vehicle depot-bike cargo after action (q_v^depot)
        vehicle_capacity    : vehicle capacity K (> 0)
        dist_to_depot       : travel time from vehicle to nearest depot (minutes)
        maintenance_enabled : whether to include maintenance features

    Returns:
        φ  np.ndarray of shape (N_FEATURES,)  dtype float64
    """
    K = max(vehicle_capacity, 1)
    N = max(len(func), 1)   # number of stations – used to normalise sum-over-stations features

    # ── Category A (Rebalancing) ──────────────────────────────────────────
    # ── φ_1  Rebalancing imbalance:   Σ_i |I_i^func − Î_i^func| / N
    phi_1 = float(np.sum(np.abs(func - target))) / N

    # ──  vehicle_functional_load: q_v^func / K
    phi_func_load = func_cargo_veh / K

    # ── demand_weighted_imbalance: Σ_i (|I_i^func - Î_i^func| × λ_i) / N
    phi_dwi = float(np.dot(np.abs(func - target), activity)) / N

    if not maintenance_enabled:
        return np.array([phi_1, phi_func_load, phi_dwi], dtype=np.float64)

    # ── Category B (Maintenance) ──────────────────────────────────────────
    # ── trailer_cannibalization: q_v^depot / K
    phi_2 = depot_cargo_veh / K

    # ──  Global onsite backlog:   Σ_i I_i^onsite / N
    phi_3 = float(np.sum(onsite)) / N

    # ──  Demand-weighted depot backlog: Σ_i (I_i^depot × λ_i) / N
    phi_4 = float(np.dot(depot, activity)) / N

    # ──  Depot pull: φ_2 × dist(v, depot) / 30
    # Divide by 30 min ≈ typical cross-city travel time to keep O(1).
    phi_5 = phi_2 * dist_to_depot / 30.0

    return np.array([
        phi_1, phi_func_load, phi_dwi,
        phi_2, phi_3, phi_4, phi_5
    ], dtype=np.float64)


def as_dict(phi: np.ndarray, maintenance_enabled: bool = True) -> dict:
    """
    Return a labelled dict of a computed feature vector — useful for
    debugging and logging individual feature values.

    Example::

        phi = extract(...)
        print(as_dict(phi))
        # {'rebalancing_imbalance': 12.0, ...}
    """
    features = get_feature_names(maintenance_enabled)
    assert len(phi) == len(features), (
        f"phi has {len(phi)} elements but {len(features)} names are registered."
    )
    return dict(zip(features, phi.tolist()))
