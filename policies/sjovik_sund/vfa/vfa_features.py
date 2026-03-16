"""
vfa_features.py  –  Feature Vector φ(S^x) for the Time-Indexed Linear VFA

This is the CANONICAL definition of every feature used in the VFA.
To change the feature set:
  1. Edit FEATURE_NAMES (name and position).
  2. Edit the corresponding computation block in extract().
  3. Update the return array in extract() to match.
  4. Retrain – nothing else needs touching.

LinearVFAPolicy imports this module as the single source of truth via:
    - FEATURE_NAMES / N_FEATURES
    - extract(...)
    - as_dict(...)

This module is pure numpy with no simulator dependencies.
All simulator-specific preparation (inventory arrays, caches) is done by
LinearVFAPolicy before calling extract().

──────────────────────────────────────────────────────────────────────────────
Notation
────────
  I_i^func   functional bikes at station i  (post-decision)
  Î_i^func   time-indexed target inventory at station i
  I_i^onsite onsite-repairable bikes at station i
  I_i^depot  depot-level damaged bikes at station i
  λ_i        time-averaged arrival rate at station i
  q_v^depot  depot-damaged bikes on the vehicle (post-decision)
  K          vehicle capacity
  dist(v,d)  travel time from vehicle location to nearest depot (minutes)

Features
────────
  φ_1  rebalancing_imbalance   Σ_i |I_i^func − Î_i^func| / N
  φ_2  trailer_cannibalization q_v^depot / K
  φ_3  onsite_backlog          Σ_i I_i^onsite / N
  φ_4  demand_weighted_depot   Σ_i (I_i^depot × λ_i) / N
  φ_5  depot_pull              φ_2 × dist(v, depot) / 30
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from typing import List


# ── Feature registry ───────────────────────────────────────────────────────────────
FEATURE_NAMES: List[str] = [
    "rebalancing_imbalance",    # φ_1
    "trailer_cannibalization",  # φ_2
    "onsite_backlog",           # φ_3
    "demand_weighted_depot",    # φ_4
    "depot_pull",               # φ_5
]
N_FEATURES: int = len(FEATURE_NAMES)   # auto-derived; never edit manually


def extract(
    func:             np.ndarray,   # (N,) functional bikes post-decision
    onsite:           np.ndarray,   # (N,) onsite-repairable bikes
    depot:            np.ndarray,   # (N,) depot-level damaged bikes
    target:           np.ndarray,   # (N,) time-indexed target Î_i^func
    activity:         np.ndarray,   # (N,) time-averaged arrival rate λ_i
    depot_cargo_veh:  float,        # q_v^depot  (post-decision)
    vehicle_capacity: int,          # K
    dist_to_depot:    float,        # dist(v, depot) in simulation minutes
) -> np.ndarray:
    """
    Compute φ(S^x) from pre-resolved inventory arrays.

    Pure numpy – no simulator calls, no side-effects.
    All inputs are prepared by LinearVFAPolicy before calling this function.

    Args:
        func             : (N,) functional bike counts (post-decision)
        onsite           : (N,) onsite-repairable bike counts
        depot            : (N,) depot-level damaged bike counts
        target           : (N,) time-indexed target inventory Î_i^func
        activity         : (N,) time-averaged arrival rate λ_i per station
        depot_cargo_veh  : vehicle depot-bike cargo after action (q_v^depot)
        vehicle_capacity : vehicle capacity K (> 0)
        dist_to_depot    : travel time from vehicle to nearest depot (minutes)

    Returns:
        φ  np.ndarray of shape (N_FEATURES,)  dtype float64
    """
    K = max(vehicle_capacity, 1)
    N = max(len(func), 1)   # number of stations – used to normalise sum-over-stations features

    # ── φ_1  Rebalancing imbalance:   Σ_i |I_i^func − Î_i^func| / N ─────────
    phi_1 = float(np.sum(np.abs(func - target))) / N

    # ── φ_2  Trailer cannibalization: q_v^depot / K ────────────────────────
    phi_2 = depot_cargo_veh / K

    # ── φ_3  Global onsite backlog:   Σ_i I_i^onsite / N ──────────────────
    phi_3 = float(np.sum(onsite)) / N

    # ── φ_4  Demand-weighted depot backlog: Σ_i (I_i^depot × λ_i) / N ────
    phi_4 = float(np.dot(depot, activity)) / N

    # ── φ_5  Depot pull: φ_2 × dist(v, depot) / 30 ──────────────────────
    # Divide by 30 min ≈ typical cross-city travel time to keep O(1).
    phi_5 = phi_2 * dist_to_depot / 30.0

    # ── Assemble & validate ───────────────────────────────────────────────
    phi = np.array([phi_1, phi_2, phi_3, phi_4, phi_5], dtype=np.float64)

    assert len(phi) == N_FEATURES, (
        f"Feature count mismatch: extract() returns {len(phi)} values but "
        f"FEATURE_NAMES has {N_FEATURES} entries. "
        f"Keep the return array and FEATURE_NAMES in sync."
    )

    return phi


def as_dict(phi: np.ndarray) -> dict:
    """
    Return a labelled dict of a computed feature vector — useful for
    debugging and logging individual feature values.

    Example::

        phi = extract(...)
        print(as_dict(phi))
        # {'rebalancing_imbalance': 12.0, 'trailer_cannibalization': 0.4, ...}
    """
    assert len(phi) == N_FEATURES, (
        f"phi has {len(phi)} elements but {N_FEATURES} names are registered."
    )
    return dict(zip(FEATURE_NAMES, phi.tolist()))
