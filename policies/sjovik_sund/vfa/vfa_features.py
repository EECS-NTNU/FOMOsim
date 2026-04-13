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
  A2  anticipated_demand_shortfall  Σ_i [max(0,outflow_i-I_i) + max(0,I_i+inflow_i-C_i)] / (0.5*Σ C_i)
  A3  squared_starvation_penalty    (1/N) Σ_i (max(0, T_i-I_i) / T_i)^2
  A4  squared_congestion_penalty    (1/N) Σ_i (max(0, I_i-T_i) / (C_i-T_i))^2
  A5  proximity_to_demand_gravity   1 - Σ_i (|λ_i| / (d_i+1)) / Φ_max
  A6  starvation_gravity            (q_func/K) * Σ_i (outflow_i * 1[I_i<T_i] / (d_i+1)) / Φ_max
  A7  congestion_gravity            (free/K) * Σ_i (inflow_i * 1[I_i>T_i] / (d_i+1)) / Φ_max
  A8  imbalance_weighted_distance   Σ_i |I_i-T_i| * d_i / (0.5*Σ C_i * max_d)
  A9  starvation_severity_max       max_i (max(0, T_i-I_i) / T_i)
  A10 congestion_severity_max       max_i (max(0, I_i-T_i) / (C_i-T_i))
  A11 station_starvation_count      Σ_i 1[I_i < T_i] / N
  A12 work_ratio                    Σ_i |I_i-T_i| / (4*K)     ← van trips needed
  A13 imbalance_concentration       max_i |I_i-T_i| / Σ_i |I_i-T_i|  ← tractability

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

Category D  —  Temporal Demand  (appended if temporal_enabled)
──────────────────────────────────────────────────────────────────────────────
  D1  time_of_day_fraction          current_hour / 24
  D2  day_of_week_fraction          current_day / 7
  D3  hours_until_peak_fraction     hours_until_8am / 12
  D4  multi_horizon_starvation_risk Σ_{h=1..H} Σ_i max(0, T_i^{t+h} - I_i) / (H * 0.5*Σ C_i)
  D5  temporal_demand_gradient      (Σ_i T_i^{t+1} - Σ_i T_i) / (0.5*Σ C_i)
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
        "anticipated_demand_shortfall",   # A2
        "squared_starvation_penalty",     # A3
        "squared_congestion_penalty",     # A4
        "proximity_to_demand_gravity",    # A5
        "starvation_gravity",             # A6
        "congestion_gravity",             # A7
        "imbalance_weighted_distance",    # A8
        "starvation_severity_max",        # A9
        "congestion_severity_max",        # A10
        "station_starvation_count",       # A11
        "work_ratio",                     # A12
        "imbalance_concentration",        # A13
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
        ])

    # Category D: Temporal Demand
    if temporal_enabled:
        names.extend([
            "time_of_day_fraction",           # D1
            "day_of_week_fraction",           # D2
            "hours_until_peak_fraction",      # D3
            "multi_horizon_starvation_risk",  # D4
            "temporal_demand_gradient",       # D5
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
    outflow = np.maximum(0.0, -activity)   # net rentals  (negative activity)
    inflow  = np.maximum(0.0,  activity)   # net returns  (positive activity)

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

    # A2: Anticipated Demand Shortfall — one-step starvation + congestion risk
    # Normalised by total_cap_half (same denominator as A1 and A8) so A2 is on the same scale
    # as the other network-level imbalance features. Interpretation: "what fraction of half
    # the network capacity is currently at immediate demand risk?"
    starvation_risk = np.maximum(0.0, outflow - func)
    congestion_risk = np.maximum(0.0, func + inflow - capacities)
    phi_demand_shortfall = np.clip(
        np.sum(starvation_risk + congestion_risk) / max(total_cap_half, 1.0),
        0.0, 1.0,
    )

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

    # A5: Proximity to Demand Gravity (inverted — penalty for being far from demand)
    raw_gravity   = np.sum(np.abs(activity) / (dist_to_stations + 1.0)) / grav_safe
    phi_demand_gravity = 1.0 - np.clip(raw_gravity, 0.0, 1.0)

    # A6: Starvation Gravity — penalty for holding bikes near starving stations
    # (van load) × (demand-weighted proximity to stations below target)
    starving_mask      = func < target
    starvation_grav    = np.sum((outflow * starving_mask) / (dist_to_stations + 1.0))
    phi_starvation_grav = np.clip(veh_load * (starvation_grav / grav_safe), 0.0, 1.0)

    # ── Debug: gravity masking (~1% of calls) ─────────────────────────────────
    if np.random.rand() < 0.01 and starvation_grav > 0.5 and phi_starvation_grav == 0.0:
        print(f"\n[DEBUG - GRAVITY MASKING] starvation_grav={starvation_grav:.2f} but A6=0.")
        print(f"  veh_load={veh_load:.2f}  (func_cargo_veh={func_cargo_veh})")
        if veh_load == 0.0:
            print("  -> Confirmed: van is empty, so A6 is correctly masked.")

    # A7: Congestion Gravity — penalty for an empty van near congested stations
    # (van free space) × (return-weighted proximity to stations above target)
    congested_mask     = func > target
    congestion_grav    = np.sum((inflow * congested_mask) / (dist_to_stations + 1.0))
    phi_congestion_grav = np.clip(veh_free_frac * (congestion_grav / grav_safe), 0.0, 1.0)

    # A8: Imbalance Weighted Distance — recoverability (distant imbalance = deferred cost)
    max_dist_safe      = max(float(np.max(dist_to_stations)), 1.0)
    phi_imbalance_dist = np.clip(
        np.sum(np.abs(func - target) * dist_to_stations) / (total_cap_half * max_dist_safe),
        0.0, 1.0,
    )

    # A9: Starvation Severity Max — worst single-station starvation ratio
    # 1.0 = at least one station has zero bikes against a non-zero target.
    phi_starvation_max = float(np.max(starv_ratio))

    # A10: Congestion Severity Max — worst single-station congestion ratio (symmetric to A9)
    phi_congestion_max = float(np.max(cong_ratio))

    # A11: Station Starvation Count — breadth of starvation (how many stations, not how much)
    phi_starvation_cnt = float(np.sum(func < target)) / N

    # A12: Work Ratio — total van trips needed to fix all imbalance
    # 4 full one-way loads → 1.0 (barely recoverable). Key long-term rollout signal.
    phi_work_ratio = np.clip(total_imbalance / max(4.0 * K_safe, 1.0), 0.0, 1.0)

    # A13: Imbalance Concentration — tractability of remaining work
    # High (→1): one station dominates; fixable with a single visit.
    # Low (→1/N): spread evenly; many trips needed regardless of van state.
    phi_concentration = (
        float(np.max(np.abs(func - target))) / total_imbalance
        if total_imbalance >= 1e-6 else 0.0
    )

    cat_a = np.clip(
        [phi_imbalance, phi_demand_shortfall, phi_starvation_sq, phi_congestion_sq,
         phi_demand_gravity, phi_starvation_grav, phi_congestion_grav,
         phi_imbalance_dist, phi_starvation_max, phi_congestion_max,
         phi_starvation_cnt, phi_work_ratio, phi_concentration],
        0.0, 1.0,
    ).tolist()
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

        cat_b = np.clip(
            [phi_cannibalization, phi_onsite_backlog, phi_depot_backlog,
             phi_depot_pull, phi_maint_urgency],
            0.0, 1.0,
        ).tolist()
        features.extend(cat_b)

    # =========================================================================
    # Category C: End-of-Day Timing Features
    # =========================================================================
    if shift_timing_enabled:
        if time_remaining is None:
            time_remaining = shift_length
        shift_length_safe = max(shift_length, 1.0)

        # C1: Time Remaining Fraction ∈ [0, 1]
        phi_time_remaining = np.clip(float(time_remaining) / shift_length_safe, 0.0, 1.0)

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

        cat_c = np.clip([phi_time_remaining, phi_time_penalty, phi_reachable], 0.0, 1.0).tolist()
        features.extend(cat_c)

    # =========================================================================
    # Category D: Temporal Demand Features
    # =========================================================================
    if temporal_enabled:
        current_hour              = int((current_time_minutes // 60) % 24)
        hours_until_morning_peak  = (_MORNING_PEAK_HOUR - current_hour) % 24

        # D1: Time of Day Fraction — where in the 24h cycle (0 = midnight, 0.5 = noon)
        phi_time_of_day = current_hour / 24.0

        # D2: Day of Week Fraction — where in the weekly cycle (0 = Monday)
        phi_day_of_week = current_day_of_week / 7.0

        # D3: Hours Until Peak Fraction — proximity to next morning rush (0 = at peak, 1 = 12h away)
        phi_peak_distance = np.clip(hours_until_morning_peak / 12.0, 0.0, 1.0)

        # D4: Multi-Horizon Starvation Risk — integrated starvation over next H hours
        if target_matrix is not None:
            abs_hour_now      = int(current_time_minutes // 60)
            total_future_risk = 0.0
            for h in range(1, horizon_hours + 1):
                abs_h       = abs_hour_now + h
                future_hour = abs_h % 24
                future_day  = (current_day_of_week + abs_h // 24) % 7
                total_future_risk += np.sum(np.maximum(0.0, target_matrix[future_day, future_hour] - func))
            phi_horizon_risk = np.clip(
                total_future_risk / (horizon_hours * max(total_cap_half, 1.0)),
                0.0, 1.0,
            )
        else:
            phi_horizon_risk = 0.0

        # D5: Temporal Demand Gradient — is total network target rising or falling next hour?
        # Positive = more bikes needed soon.  Range: [-1, 1].
        if target_matrix is not None:
            abs_hour_next   = int(current_time_minutes // 60) + 1
            next_hour       = abs_hour_next % 24
            next_day        = (current_day_of_week + abs_hour_next // 24) % 7
            phi_demand_grad = np.clip(
                (float(np.sum(target_matrix[next_day, next_hour])) - float(np.sum(target)))
                / max(total_cap_half, 1.0),
                -1.0, 1.0,
            )
        else:
            phi_demand_grad = 0.0

        cat_d = [phi_time_of_day, phi_day_of_week, phi_peak_distance,
                 phi_horizon_risk, phi_demand_grad]
        features.extend(cat_d)

    return np.array(features, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

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
