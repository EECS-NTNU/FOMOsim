"""
nn_state_encoder.py  —  MDPState → Tensor Encoder

This module is the exclusive boundary between the MDP layer and the neural
network. Its sole job is to convert an MDPState snapshot into three tensors
that the NNValueNetwork consumes:

    station_block   [N_stations  × STATION_FEATURE_DIM]
    vehicle_block   [M_vehicles  × VEHICLE_FEATURE_DIM]
    global_context  [GLOBAL_FEATURE_DIM]

─────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS (AND WHY IT DIFFERS FROM vfa_features.py)
─────────────────────────────────────────────────────────────────────────────
The linear VFA in vfa_features.py manually engineers composite signals:
gravity terms, imbalance products, demand-weighted distances, etc. Those
features encode domain knowledge about *how* to think about the state.

This encoder deliberately avoids that. It only normalizes raw quantities
(counts → ratios, time → cyclic encoding) and passes them through. The
neural network is expected to discover the composite relationships itself.
The less we preprocess, the fewer assumptions we bake in.

DESIGN CONTRACTS:
  - Input:  MDPState (policies/sjovik_sund/mdp/mdp_formulation.py)
  - Output: dict["station_block", "vehicle_block", "global_context"]
  - Zero dependency on vfa_features.py or any VFA-specific logic
  - No live simulator objects — only the already-extracted MDPState
  - Unit-testable without running a simulation

─────────────────────────────────────────────────────────────────────────────
FLOW OVERVIEW
─────────────────────────────────────────────────────────────────────────────

  MDPState
      │
      ├─ mdp_state.stations  ──►  encode_station_block()  ──►  [N × 8]
      │       (N StationInventory objects, sorted by ID for determinism)
      │
      ├─ mdp_state.vehicles  ──►  encode_vehicle_block()  ──►  [M × 5]
      │       (M VehicleStatus objects, sorted by ID)
      │
      └─ mdp_state (whole)   ──►  encode_global_context() ──►  [7]
              (time, system pressure, depot, shift remaining, fleet load)
              │
              └──► combined via encode_state() ──► {"station_block", ...}
"""

import math
import torch
from typing import Dict, Optional

from policies.sjovik_sund.mdp.mdp_formulation import MDPState, StationInventory, VehicleStatus
from settings import SERVICE_TIME_FROM, SERVICE_TIME_TO

_SHIFT_START_MIN  = SERVICE_TIME_FROM * 60                          # 420
_SHIFT_END_MIN    = SERVICE_TIME_TO   * 60                          # 1200
_SHIFT_LENGTH_MIN = _SHIFT_END_MIN - _SHIFT_START_MIN              # 780


# ─────────────────────────────────────────────────────────────────────────────
# Dimension constants — imported by nn_model.py to keep shapes in sync
# ─────────────────────────────────────────────────────────────────────────────

STATION_FEATURE_DIM = 10  # base features per station row
VEHICLE_FEATURE_DIM = 6   # features per vehicle row  (func_cargo, depot_cargo, dest_func, eta, dest_id)
GLOBAL_FEATURE_DIM  = 8   # base entries in the global context vector
ACTION_CONTEXT_DIM  = 10  # appended to global context when action-context encoding is enabled
DEMAND_HORIZON_DIM  = 4   # appended to each station row when demand-horizon encoding is enabled
GLOBAL_HEALTH_DIM   = 8   # appended to global context when global-health encoding is enabled

_USE_ACTION_CONTEXT = False
_USE_DEMAND_HORIZON = False
_USE_GLOBAL_HEALTH = False


def set_encoder_options(
    use_action_context: bool = False,
    use_demand_horizon: bool = False,
    use_global_health: bool = False,
) -> None:
    """Set process-local encoder options used by training/evaluation."""
    global _USE_ACTION_CONTEXT, _USE_DEMAND_HORIZON, _USE_GLOBAL_HEALTH
    _USE_ACTION_CONTEXT = bool(use_action_context)
    _USE_DEMAND_HORIZON = bool(use_demand_horizon)
    _USE_GLOBAL_HEALTH = bool(use_global_health)


def get_global_feature_dim(use_action_context: bool = None, use_global_health: bool = None) -> int:
    if use_action_context is None:
        use_action_context = _USE_ACTION_CONTEXT
    if use_global_health is None:
        use_global_health = _USE_GLOBAL_HEALTH
    return (
        GLOBAL_FEATURE_DIM
        + (GLOBAL_HEALTH_DIM if use_global_health else 0)
        + (ACTION_CONTEXT_DIM if use_action_context else 0)
    )


def get_station_feature_dim(use_demand_horizon: bool = None) -> int:
    if use_demand_horizon is None:
        use_demand_horizon = _USE_DEMAND_HORIZON
    return STATION_FEATURE_DIM + (DEMAND_HORIZON_DIM if use_demand_horizon else 0)

# ═════════════════════════════════════════════════════════════════════════════
# STATION BLOCK  [N_stations × STATION_FEATURE_DIM]
# ═════════════════════════════════════════════════════════════════════════════

def _late_shift_weight(time_minutes: float) -> float:
    time_of_day = time_minutes % 1440.0
    minutes_to_end = _SHIFT_END_MIN - time_of_day
    if minutes_to_end < 0:
        return 1.0
    late_window = 2.0 * 60.0
    return min(1.0, max(0.0, 1.0 - minutes_to_end / late_window))


def _encode_demand_horizon(inv: StationInventory, time_minutes: float) -> list:
    cap = inv.capacity if inv.capacity > 0 else 1
    net_flow = inv.expected_arrival_rate - inv.expected_departure_rate
    projected_1h = inv.functional + net_flow
    projected_2h = inv.functional + 2.0 * net_flow

    overflow = max(0.0, projected_2h - cap)
    empty = max(0.0, -projected_2h)
    violation_risk = min(1.0, max(overflow, empty) / cap)

    # MDPState only carries the current target, not tomorrow's full target
    # schedule. This is a late-shift proxy for the next-morning positioning gap.
    late_target_gap = ((inv.target - inv.functional) / cap) * _late_shift_weight(time_minutes)

    def _clip(x: float, lo: float = -2.0, hi: float = 2.0) -> float:
        return max(lo, min(hi, x))

    return [
        _clip(net_flow / cap),
        _clip((2.0 * net_flow) / cap),
        violation_risk,
        max(-1.0, min(1.0, late_target_gap)),
    ]


def _encode_station(inv: StationInventory, eta_from_dest: float, is_destination: bool, time_minutes: float) -> list:
    """
    Encode one StationInventory as a 9-element feature vector.

    All quantities are normalized by station capacity so the values are in
    [0, 1] regardless of how large or small the station is.  This lets the
    shared station encoder generalize across stations of different sizes.

    Features (index → meaning):
      [0] functional_ratio  : rentable bikes / capacity
      [1] onsite_ratio      : bikes repairable on-site / capacity
      [2] depot_ratio       : bikes requiring depot removal / capacity
      [3] eta_from_dest     : travel time from the vehicle's DESTINATION to THIS
                              station, normalized by 60 min.
                              Using destination (not current position) means each
                              candidate action produces a distinct station block,
                              enabling the NN to learn gravity-like spatial patterns
                              (e.g. "how reachable are starved stations from here?").
      [4] target_ratio      : target inventory / capacity — optimal fill level
                              for this station at the current time-of-day.
      [5] deficit_ratio     : (target - functional) / capacity — signed imbalance.
                              > 0: station is starving, < 0: station is congested.
      [6] departure_rate    : expected_departure_rate / capacity — capped at 2.0.
      [7] arrival_rate      : expected_arrival_rate / capacity — capped at 2.0.
      [8] net_flow_ratio  : (arrival_rate - departure_rate) / capacity, clamped [-2, 2].
                              >0: inventory growing (heading toward congestion),
                              <0: inventory shrinking (heading toward starvation).
                              The NN could compute this from [6] and [7], but providing
                              it directly removes the need to learn the subtraction.
      [9] is_destination    : 1.0 if this station is the vehicle's next destination,
                              0.0 otherwise. Allows cross-attention to explicitly
                              focus on the chosen destination station.
    """
    cap = inv.capacity if inv.capacity > 0 else 1

    target_ratio   = inv.target / cap
    deficit_ratio  = (inv.target - inv.functional) / cap
    departure_rate = min(2.0, inv.expected_departure_rate / cap)
    arrival_rate   = min(2.0, inv.expected_arrival_rate   / cap)
    net_flow_ratio = max(-2.0, min(2.0, (inv.expected_arrival_rate - inv.expected_departure_rate) / cap))

    features = [
        inv.functional / cap,   # [0] functional_ratio
        inv.onsite     / cap,   # [1] onsite_ratio
        inv.depot      / cap,   # [2] depot_ratio
        eta_from_dest,          # [3] travel time from destination (normalized by 60 min)
        target_ratio,           # [4] optimal fill level at current time-of-day
        deficit_ratio,          # [5] signed gap: >0 starving, <0 congested
        departure_rate,         # [6] expected outflow rate / capacity
        arrival_rate,           # [7] expected inflow rate / capacity
        net_flow_ratio,         # [8] net inventory trend: >0 filling, <0 draining
        float(is_destination),  # [9] 1.0 if this is the vehicle's next destination
    ]
    if _USE_DEMAND_HORIZON:
        features.extend(_encode_demand_horizon(inv, time_minutes))
    return features


def encode_station_block(mdp_state: MDPState, dest_travel_times: dict = None) -> torch.Tensor:
    """
    Build the station feature matrix.

    Args:
        mdp_state         : MDPState at the current decision epoch.
        dest_travel_times : {station_id: minutes} from the vehicle's DESTINATION.
                            If provided, feature [3] encodes distances from the
                            destination (unique per candidate action). Falls back
                            to mdp_state.travel_times (from current position) when
                            None — used for terminal-state evaluation where the
                            vehicle has already arrived.

    Returns:
        Float32 tensor of shape [N_stations, STATION_FEATURE_DIM].
    """
    tt = dest_travel_times if dest_travel_times is not None else (mdp_state.travel_times or {})

    destination_station = None
    if mdp_state.active_vehicle_id is not None:
        v = mdp_state.vehicles.get(mdp_state.active_vehicle_id)
        if v is not None:
            destination_station = v.destination_station

    rows = []
    for sid in sorted(mdp_state.stations.keys()):
        raw_tt = tt.get(sid, 0.0)
        eta = min(1.0, raw_tt / 60.0)
        is_dest = (sid == destination_station)
        rows.append(_encode_station(mdp_state.stations[sid], eta, is_dest, mdp_state.time))

    return torch.tensor(rows, dtype=torch.float32)   # [N, STATION_FEATURE_DIM]


# ═════════════════════════════════════════════════════════════════════════════
# VEHICLE BLOCK  [M_vehicles × VEHICLE_FEATURE_DIM]
# ═════════════════════════════════════════════════════════════════════════════

    
def _encode_vehicle(
    status: VehicleStatus,
    stations: dict,
    shift_end_time: Optional[float],
    current_time: float,
    # sorted_station_ids is no longer needed
) -> list:
    
    cap = status.capacity if status.capacity > 0 else 1
    functional_cargo_ratio = status.functional_cargo / cap
    depot_cargo_ratio      = status.depot_cargo      / cap

    # --- Destination station's full context ---
    dest_inv = stations.get(status.destination_station)
    if dest_inv is not None:
        dest_cap              = dest_inv.capacity if dest_inv.capacity > 0 else 1
        dest_functional_ratio = dest_inv.functional / dest_cap
        dest_target_ratio     = dest_inv.target / dest_cap
        dest_deficit_ratio    = (dest_inv.target - dest_inv.functional) / dest_cap
    else:
        # Depot or unknown destination fallback
        dest_functional_ratio = 0.0
        dest_target_ratio     = 0.0
        dest_deficit_ratio    = 0.0

    # --- Normalized ETA (Fixed denominator to 60.0 instead of 1440.0) ---
    time_until_arrival = max(0.0, status.eta - current_time)
    eta_normalized = min(1.0, time_until_arrival / 60.0)

    return [
        functional_cargo_ratio,   # [0] 
        depot_cargo_ratio,        # [1] 
        dest_functional_ratio,    # [2] Destination current fill
        dest_target_ratio,        # [3] NEW: Destination optimal fill
        dest_deficit_ratio,       # [4] NEW: Destination starvation/congestion gap
        eta_normalized,           # [5] Fixed ETA scale
    ]


def encode_vehicle_block(mdp_state: MDPState) -> torch.Tensor:
    """
    Build the vehicle feature matrix.

    Vehicles are sorted by vehicle ID for determinism.  The same
    permutation-invariance argument as for stations applies here.

    Args:
        mdp_state : MDPState at the current decision epoch.

    Returns:
        Float32 tensor of shape [M_vehicles, VEHICLE_FEATURE_DIM].
    """
    #sorted_station_ids = sorted(mdp_state.stations.keys())
    rows = [
        _encode_vehicle(
            mdp_state.vehicles[vid],
            mdp_state.stations,
            mdp_state.shift_end_time,
            mdp_state.time,
            #sorted_station_ids,
        )
        for vid in sorted(mdp_state.vehicles.keys())
    ]
    return torch.tensor(rows, dtype=torch.float32)   # [M, VEHICLE_FEATURE_DIM]


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CONTEXT VECTOR  [GLOBAL_FEATURE_DIM]
# ═════════════════════════════════════════════════════════════════════════════

def encode_global_context(mdp_state: MDPState) -> torch.Tensor:
    """
    Build the global context vector that captures system-wide and temporal
    information not visible from any single station or vehicle in isolation.

    Features (index → meaning):
      [0] time_sin         : sin(2π · hour_of_day / 24)  ─┐ cyclic time
      [1] time_cos         : cos(2π · hour_of_day / 24)  ─┘ encoding

                             WHY CYCLIC? A raw hour/24 jumps from ~1 at
                             23:59 to 0 at 00:00.  The (sin, cos) pair
                             maps time onto the unit circle so the model
                             sees a smooth, periodic signal and can learn
                             demand patterns that repeat daily.

      [2] starved_ratio     : fraction of stations with 0 functional bikes.
      [3] low_ratio         : fraction of stations with functional < 10% capacity.
                             Together these give the NN a coarse demand-pressure
                             signal: fully empty vs. critically low.

      [3] total_broken     : total broken bikes (onsite + depot) across
                             all stations, normalized by total capacity.
                             Global maintenance backlog signal.

      [4] depot_queue_ratio  (shifted +1 due to low_ratio insertion): fixed_queue / (fixed_queue + in_repair).
                             0 = all depot bikes still in repair (no
                             stock available for pickup),
                             1 = all repair finished (ready to reload).

      [5] shift_remaining  : (1440 - time % 1440) / 1440. Fraction of the
                             24-hour day remaining. Enables anticipatory
                             end-of-day behavior (e.g., head toward depot
                             before shift ends). Always in (0, 1].

      [6] mean_load_ratio  : mean functional_cargo / capacity across fleet.
                             Whether vehicles are collectively loaded or empty.

    Args:
        mdp_state : MDPState at the current decision epoch.

    Returns:
        Float32 tensor of shape [GLOBAL_FEATURE_DIM].
    """
    # --- cyclic time-of-day encoding ---
    hour_of_day = (mdp_state.time % (24 * 60)) / 60.0   # simulation minutes → hour
    time_sin = math.sin(2 * math.pi * hour_of_day / 24)
    time_cos = math.cos(2 * math.pi * hour_of_day / 24)

    # --- system-wide starvation signals ---
    n_stations  = max(len(mdp_state.stations), 1)
    n_starved   = sum(1 for inv in mdp_state.stations.values() if inv.functional == 0)
    n_low       = sum(
        1 for inv in mdp_state.stations.values()
        if inv.functional < 0.1 * max(inv.capacity, 1)
    )
    starved_ratio = n_starved / n_stations
    low_ratio     = n_low     / n_stations

    # --- system-wide maintenance backlog ---
    total_capacity = sum(inv.capacity for inv in mdp_state.stations.values())
    total_broken   = sum(inv.onsite + inv.depot for inv in mdp_state.stations.values())
    broken_ratio   = total_broken / max(total_capacity, 1)

    # --- depot readiness: what fraction of depot bikes are already repaired? ---
    if mdp_state.depot is not None:
        all_depot_bikes  = mdp_state.depot.fixed_queue + mdp_state.depot.in_repair
        depot_queue_ratio = mdp_state.depot.fixed_queue / max(all_depot_bikes, 1)
    else:
        depot_queue_ratio = 0.0   # no depot in this instance

    # --- normalized time remaining in shift ---
    # Normalize against the standard 24-hour shift length (1440 min), which
    # matches LinearVFAPolicy._get_shift_length() and gives a consistent [0, 1]
    # scale regardless of what absolute time shift_end_time holds.
    #
    # Previous bug: denominator was shift_end_time (an absolute timestamp),
    # making the ratio near-zero for any episode past minute ~1440.
    # Shift runs SERVICE_TIME_FROM–SERVICE_TIME_TO each day (7:00–20:00 = 780 min).
    time_of_day     = mdp_state.time % 1440.0
    shift_remaining = min(1.0, max(0.0, (_SHIFT_END_MIN - time_of_day) / _SHIFT_LENGTH_MIN))

    # --- mean functional cargo ratio across fleet ---
    if mdp_state.vehicles:
        mean_load = sum(
            v.functional_cargo / max(v.capacity, 1)
            for v in mdp_state.vehicles.values()
        ) / len(mdp_state.vehicles)
    else:
        mean_load = 0.0

    features = [time_sin, time_cos, starved_ratio, low_ratio, broken_ratio,
                depot_queue_ratio, shift_remaining, mean_load]

    if _USE_GLOBAL_HEALTH:
        total_functional = sum(inv.functional for inv in mdp_state.stations.values())
        deviations = [
            (inv.target - inv.functional) / max(inv.capacity, 1)
            for inv in mdp_state.stations.values()
        ]
        max_starving = max([d for d in deviations if d > 0.0] or [0.0])
        max_congested = max([-d for d in deviations if d < 0.0] or [0.0])
        mean_abs_dev = sum(abs(d) for d in deviations) / max(len(deviations), 1)

        total_vehicle_capacity = sum(max(v.capacity, 1) for v in mdp_state.vehicles.values())
        total_depot_cargo = sum(v.depot_cargo for v in mdp_state.vehicles.values())
        total_free_capacity = sum(max(0, v.capacity - v.functional_cargo - v.depot_cargo)
                                  for v in mdp_state.vehicles.values())

        if mdp_state.depot is not None:
            fixed_queue = mdp_state.depot.fixed_queue
            in_repair = mdp_state.depot.in_repair
            depot_den = max(total_capacity, 1)
        else:
            fixed_queue = 0
            in_repair = 0
            depot_den = max(total_capacity, 1)

        features.extend([
            total_functional / max(total_capacity, 1),
            mean_abs_dev,
            min(1.0, max_starving),
            min(1.0, max_congested),
            total_depot_cargo / max(total_vehicle_capacity, 1),
            fixed_queue / depot_den,
            in_repair / depot_den,
            total_free_capacity / max(total_vehicle_capacity, 1),
        ])

    return torch.tensor(features, dtype=torch.float32)


def encode_action_context(mdp_state: MDPState, mdp_action=None, action_duration: float = 0.0) -> torch.Tensor:
    """
    Encode the proposed action alongside the post-decision state.

    This is optional because older checkpoints were trained from state-only
    encodings. Values are normalized to roughly [-1, 1] or [0, 1].
    """
    if mdp_action is None:
        return torch.zeros(ACTION_CONTEXT_DIM, dtype=torch.float32)

    v = mdp_state.vehicles.get(mdp_state.active_vehicle_id) if mdp_state.active_vehicle_id else None
    cap = max(getattr(v, "capacity", 1), 1)
    next_station = getattr(mdp_action, "next_station", None)
    dest_inv = mdp_state.stations.get(next_station)

    if dest_inv is not None:
        dest_cap = max(dest_inv.capacity, 1)
        dest_deficit = (dest_inv.target - dest_inv.functional) / dest_cap
        dest_depot = dest_inv.depot / dest_cap
    else:
        dest_deficit = 0.0
        dest_depot = 0.0

    is_depot = 1.0 if (mdp_state.depot is not None and next_station == mdp_state.depot.station_id) else 0.0
    travel_to_next = 0.0
    if mdp_state.travel_times:
        travel_to_next = float(mdp_state.travel_times.get(next_station, 0.0) or 0.0)

    def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    return torch.tensor(
        [
            _clip(float(getattr(mdp_action, "rebalancing", 0)) / cap),
            _clip(float(getattr(mdp_action, "onsite_repairs", 0)) / cap, 0.0, 1.0),
            _clip(float(getattr(mdp_action, "depot_removals", 0)) / cap, 0.0, 1.0),
            _clip(float(getattr(mdp_action, "depot_dropoffs", 0)) / cap, 0.0, 1.0),
            _clip(float(getattr(mdp_action, "load_from_queue", 0)) / cap, 0.0, 1.0),
            is_depot,
            _clip(dest_deficit),
            _clip(dest_depot, 0.0, 1.0),
            min(1.0, travel_to_next / 60.0),
            min(1.0, max(0.0, float(action_duration or 0.0)) / 60.0),
        ],
        dtype=torch.float32,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

def encode_state(
    mdp_state: MDPState,
    dest_travel_times: dict = None,
    mdp_action=None,
    action_duration: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Convert an MDPState into the three tensor blocks consumed by NNValueNetwork.

    Args:
        mdp_state         : MDPState snapshot at the current decision epoch.
        dest_travel_times : optional {station_id: minutes} from the vehicle's
                            destination — passed through to encode_station_block.
                            Pass None for terminal-state evaluation.

    Returns:
        dict with three float32 tensors.
    """
    global_context = encode_global_context(mdp_state)
    if _USE_ACTION_CONTEXT:
        global_context = torch.cat([
            global_context,
            encode_action_context(mdp_state, mdp_action=mdp_action, action_duration=action_duration),
        ], dim=0)

    return {
        "station_block":  encode_station_block(mdp_state, dest_travel_times),
        "vehicle_block":  encode_vehicle_block(mdp_state),
        "global_context": global_context,
    }


# ═════════════════════════════════════════════════════════════════════════════
# VFA FEATURES MODE  — flat hand-crafted feature vector
# ═════════════════════════════════════════════════════════════════════════════
# Drop-in replacement for encode_state() when USE_VFA_FEATURES=True in
# train_nn_rollout.py.  Returns the same dict shape so _compute_td_loss
# and NNLearningPolicy need zero changes.  station_block and vehicle_block
# are dummy [1,1] zeros; global_context carries the full VFA feature vector.
# FlatNNValueNetwork (nn_model.py) ignores the dummies and reads global_context.

VFA_FEATURE_DIM = 28  # must match nn_state_encoder's vfa_features.extract() call

def encode_state_vfa(
    mdp_state: MDPState,
    dest_travel_times: dict = None,
    mdp_action=None,
    action_duration: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Encode post-decision MDPState using the hand-crafted VFA features from
    vfa_features.py — the same features LinearVFAPolicy uses.

    The signature matches encode_state() so it can be swapped in by aliasing:
        from nn_state_encoder import encode_state_vfa as encode_state
    dest_travel_times is accepted but unused (kept for interface compatibility).
    """
    import numpy as np
    from policies.sjovik_sund.vfa.vfa_features import extract as _vfa_extract

    sorted_sids = sorted(mdp_state.stations.keys())
    N = len(sorted_sids)

    func     = np.array([mdp_state.stations[s].functional            for s in sorted_sids], dtype=np.float64)
    onsite   = np.array([mdp_state.stations[s].onsite                for s in sorted_sids], dtype=np.float64)
    depot_st = np.array([mdp_state.stations[s].depot                 for s in sorted_sids], dtype=np.float64)
    target   = np.array([mdp_state.stations[s].target                for s in sorted_sids], dtype=np.float64)
    caps     = np.array([mdp_state.stations[s].capacity              for s in sorted_sids], dtype=np.float64)
    leave_a  = np.array([mdp_state.stations[s].expected_departure_rate for s in sorted_sids], dtype=np.float64)
    arrive_a = np.array([mdp_state.stations[s].expected_arrival_rate   for s in sorted_sids], dtype=np.float64)

    tt = mdp_state.travel_times or {}
    dist_to_stations = np.array([tt.get(s, 0.0) for s in sorted_sids], dtype=np.float64)

    v = mdp_state.vehicles.get(mdp_state.active_vehicle_id) if mdp_state.active_vehicle_id else None
    func_cargo  = float(v.functional_cargo) if v else 0.0
    depot_cargo = float(v.depot_cargo)      if v else 0.0
    veh_cap     = int(v.capacity)           if v else 1

    fleet_size = float(func.sum() + onsite.sum() + depot_st.sum() + func_cargo + depot_cargo)

    if mdp_state.depot is not None:
        _ir = mdp_state.depot.in_repair
        _fq = mdp_state.depot.fixed_queue
        depot_in_repair = float(len(_ir) if hasattr(_ir, '__len__') else _ir)
        depot_fixed_q   = float(len(_fq) if hasattr(_fq, '__len__') else _fq)
    else:
        depot_in_repair = 0.0
        depot_fixed_q   = 0.0

    time_of_day     = mdp_state.time % 1440.0
    shift_remaining = max(0.0, _SHIFT_END_MIN - time_of_day)
    lambda_max      = max(float(leave_a.max()), 1.0)  # proxy for lambda_max_system

    # Positional args for params 0-15 to avoid name mismatches across vfa_features versions.
    # Keyword args start at param 16 (all have defaults) — these names are stable.
    feats = _vfa_extract(
        func,               # 0  func
        onsite,             # 1  onsite
        depot_st,           # 2  depot
        target,             # 3  target
        caps,               # 4  capacities
        leave_a,            # 5  leave_activity / activity (name varies across versions)
        arrive_a,           # 6  arrive_activity (name varies; omit in old single-activity versions)
        dist_to_stations,   # 7  dist_to_stations
        func_cargo,         # 8  func_cargo_veh
        depot_cargo,        # 9  depot_cargo_veh
        veh_cap,            # 10 vehicle_capacity
        0.0,                # 11 dist_to_depot
        lambda_max,         # 12 lambda_max_system
        1.0,                # 13 max_gravity (unnormalized; only scales gravity features)
        fleet_size,         # 14 fleet_size
        N,                  # 15 total_stations
        depot_in_repair=depot_in_repair,
        depot_fixed_queue=depot_fixed_q,
        maintenance_enabled=True,
        logistics_enabled=True,   # SLC(5) gives total 28 features
        time_remaining=shift_remaining,
        shift_length=_SHIFT_LENGTH_MIN,
        demand_horizon_enabled=True,
        current_time_minutes=mdp_state.time,
        current_day_of_week=int(mdp_state.time // 1440) % 7,
    )

    feat_tensor = torch.tensor(feats, dtype=torch.float32)  # [VFA_FEATURE_DIM]
    return {
        "station_block":  torch.zeros(1, 1, dtype=torch.float32),   # dummy — FlatNNValueNetwork ignores this
        "vehicle_block":  torch.zeros(1, 1, dtype=torch.float32),   # dummy
        "global_context": feat_tensor,                               # [28] — the actual signal
    }
