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

STATION_FEATURE_DIM = 9   # features per station row (functional, onsite, depot, eta_from_dest, target_ratio, deficit_ratio, departure_rate, arrival_rate, is_destination)
VEHICLE_FEATURE_DIM = 6   # features per vehicle row  (func_cargo, depot_cargo, dest_func, eta, dest_id)
GLOBAL_FEATURE_DIM  = 8   # entries in the global context vector

# ═════════════════════════════════════════════════════════════════════════════
# STATION BLOCK  [N_stations × STATION_FEATURE_DIM]
# ═════════════════════════════════════════════════════════════════════════════

def _encode_station(inv: StationInventory, eta_from_dest: float, is_destination: bool) -> list:
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
      [8] is_destination    : 1.0 if this station is the vehicle's next destination,
                              0.0 otherwise. Allows cross-attention to explicitly
                              focus on the chosen destination station.
    """
    cap = inv.capacity if inv.capacity > 0 else 1

    target_ratio   = inv.target / cap
    deficit_ratio  = (inv.target - inv.functional) / cap
    departure_rate = min(2.0, inv.expected_departure_rate / cap)
    arrival_rate   = min(2.0, inv.expected_arrival_rate   / cap)

    return [
        inv.functional / cap,   # [0] functional_ratio
        inv.onsite     / cap,   # [1] onsite_ratio
        inv.depot      / cap,   # [2] depot_ratio
        eta_from_dest,          # [3] travel time from destination (normalized by 60 min)
        target_ratio,           # [4] optimal fill level at current time-of-day
        deficit_ratio,          # [5] signed gap: >0 starving, <0 congested
        departure_rate,         # [6] expected outflow rate / capacity
        arrival_rate,           # [7] expected inflow rate / capacity
        float(is_destination),  # [8] 1.0 if this is the vehicle's next destination
    ]


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
        rows.append(_encode_station(mdp_state.stations[sid], eta, is_dest))

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
    shift_remaining = max(0.0, (_SHIFT_END_MIN - time_of_day) / _SHIFT_LENGTH_MIN)

    # --- mean functional cargo ratio across fleet ---
    if mdp_state.vehicles:
        mean_load = sum(
            v.functional_cargo / max(v.capacity, 1)
            for v in mdp_state.vehicles.values()
        ) / len(mdp_state.vehicles)
    else:
        mean_load = 0.0

    return torch.tensor(
        [time_sin, time_cos, starved_ratio, low_ratio, broken_ratio,
         depot_queue_ratio, shift_remaining, mean_load],
        dtype=torch.float32,
    )   # [8]


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

def encode_state(mdp_state: MDPState, dest_travel_times: dict = None) -> Dict[str, torch.Tensor]:
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
    return {
        "station_block":  encode_station_block(mdp_state, dest_travel_times),
        "vehicle_block":  encode_vehicle_block(mdp_state),
        "global_context": encode_global_context(mdp_state),
    }
