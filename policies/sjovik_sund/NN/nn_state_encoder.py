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
      ├─ mdp_state.stations  ──►  encode_station_block()  ──►  [N × 3]
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

STATION_FEATURE_DIM = 6   # features per station row (functional, onsite, depot, time sin/cos, target_travel_time)
VEHICLE_FEATURE_DIM = 5   # features per vehicle row  (func_cargo, depot_cargo, dest_func, eta, dest_id)
GLOBAL_FEATURE_DIM  = 8   # entries in the global context vector

# ═════════════════════════════════════════════════════════════════════════════
# STATION BLOCK  [N_stations × STATION_FEATURE_DIM]
# ═════════════════════════════════════════════════════════════════════════════

def _encode_station(inv: StationInventory, hour_of_day: float, target_travel_time: float) -> list:
    """
    Encode one StationInventory as a 6-element feature vector.

    All quantities are normalized by station capacity so the values are in
    [0, 1] regardless of how large or small the station is.  This lets the
    shared station encoder generalize across stations of different sizes.

    Features (index → meaning):
      [0] functional_ratio  : rentable bikes / capacity
      [1] onsite_ratio      : bikes repairable on-site / capacity
      [2] depot_ratio       : bikes requiring depot removal / capacity
      [3] time_sin          : sin(2π · hour_of_day / 24) — cyclic time-of-day,
      [4] time_cos          : cos(2π · hour_of_day / 24)   shared per station so
                             the NN can learn per-station demand patterns by time.
      [5] target_travel_time: normalized travel time to this station if it is 
                              the destination of the active vehicle, else 0.0.
                              This gives the network explicit spatial distance.

    Note — empty_dock_ratio is intentionally omitted.
    free_docks = capacity - functional - onsite - depot, so
    functional_ratio + onsite_ratio + depot_ratio + empty_dock_ratio = 1 always.
    Including it adds a perfectly linearly dependent fourth column — the network
    can always derive it, and including it wastes a dimension while forcing the
    encoder weights to compensate for the constant-sum constraint.
    """
    cap = inv.capacity if inv.capacity > 0 else 1  # guard zero-capacity stations

    return [
        inv.functional / cap,                           # [0] functional_ratio
        inv.onsite     / cap,                           # [1] onsite_ratio
        inv.depot      / cap,                           # [2] depot_ratio
        math.sin(2 * math.pi * hour_of_day / 24),      # [3] time_sin
        math.cos(2 * math.pi * hour_of_day / 24),      # [4] time_cos
        target_travel_time,                            # [5] explicit travel time to destination
    ]


def encode_station_block(mdp_state: MDPState) -> torch.Tensor:
    """
    Build the station feature matrix.

    Stations are sorted by ID to guarantee a consistent layout across calls.
    Ordering does not need to be semantically meaningful: the StationEncoder
    in nn_model.py applies the same shared weights to every row, and the
    result is then mean-pooled across stations — so permutation order does
    not affect the value estimate.  Sorting is only for determinism.

    Args:
        mdp_state : MDPState at the current decision epoch.

    Returns:
        Float32 tensor of shape [N_stations, STATION_FEATURE_DIM].
    """
    hour_of_day = (mdp_state.time % (24 * 60)) / 60.0
    
    active_vehicle = mdp_state.vehicles.get(mdp_state.active_vehicle_id)
    target_station_id = None
    target_travel_time = 0.0
    
    if active_vehicle:
        target_station_id = active_vehicle.destination_station
        time_until_arrival = max(0.0, active_vehicle.eta - mdp_state.time)
        target_travel_time = min(1.0, time_until_arrival / 1440.0)

    rows = []
    for sid in sorted(mdp_state.stations.keys()):
        tt_val = target_travel_time if sid == target_station_id else 0.0
        rows.append(_encode_station(mdp_state.stations[sid], hour_of_day, tt_val))
        
    return torch.tensor(rows, dtype=torch.float32)   # [N, STATION_FEATURE_DIM]


# ═════════════════════════════════════════════════════════════════════════════
# VEHICLE BLOCK  [M_vehicles × VEHICLE_FEATURE_DIM]
# ═════════════════════════════════════════════════════════════════════════════

def _encode_vehicle(
    status: VehicleStatus,
    stations: dict,
    shift_end_time: Optional[float],
    current_time: float,
    sorted_station_ids: list,
) -> list:
    """
    Encode one VehicleStatus as a 5-element feature vector.

    Features (index → meaning):
      [0] functional_cargo_ratio   : functional bikes on board / capacity
      [1] depot_cargo_ratio        : depot-damaged bikes on board / capacity

      Note — remaining_capacity_ratio is intentionally omitted.
      free_capacity = capacity - functional_cargo - depot_cargo, so
      features [0]+[1]+[2] would always sum to 1 — a perfect linear dependency.
      The two cargo ratios already fully describe vehicle load state.

      [2] dest_functional_ratio    : functional bikes / capacity at the
                                     destination station, i.e. how full the
                                     target station currently is.
                                     Falls back to 0.0 for depot destinations
                                     or unknown station IDs.

      [3] eta_normalized           : time until arrival / shift_length (1440 min).
                                     0.0 = vehicle already arrived,
                                     ~1.0 = vehicle just departed near shift start.
                                     Falls back to a fixed 8-hour window if no
                                     shift_end_time is defined.

      [4] dest_id_normalized       : sorted index of destination station / n_stations.
                                     Gives the NN a stable identity signal so it can
                                     learn station-specific value differences.
                                     0.0 for depot or unknown destinations.
    """
    cap = status.capacity if status.capacity > 0 else 1

    # --- cargo composition (two independent values; third is 1 - these two) ---
    functional_cargo_ratio = status.functional_cargo / cap
    depot_cargo_ratio      = status.depot_cargo      / cap

    # --- destination station's current fill level ---
    dest_inv = stations.get(status.destination_station)
    if dest_inv is not None:
        dest_cap              = dest_inv.capacity if dest_inv.capacity > 0 else 1
        dest_functional_ratio = dest_inv.functional / dest_cap
    else:
        # Depot or unknown destination
        dest_functional_ratio = 0.0

    # --- normalized ETA ---
    # Normalized against the standard 24-hour shift length (1440 min), matching
    # the reference used by LinearVFAPolicy._get_shift_length().
    # This gives a consistent scale across all decision epochs.
    time_until_arrival = max(0.0, status.eta - current_time)
    eta_normalized = min(1.0, time_until_arrival / 1440.0)

    # --- normalized destination station identity ---
    n_stations = max(len(sorted_station_ids), 1)
    if status.destination_station in stations:
        dest_idx = sorted_station_ids.index(status.destination_station)
        dest_id_normalized = dest_idx / n_stations
    else:
        dest_id_normalized = 0.0   # depot or unknown

    return [
        functional_cargo_ratio,   # [0]
        depot_cargo_ratio,        # [1]
        dest_functional_ratio,    # [2]
        eta_normalized,           # [3]
        dest_id_normalized,       # [4]
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
    sorted_station_ids = sorted(mdp_state.stations.keys())
    rows = [
        _encode_vehicle(
            mdp_state.vehicles[vid],
            mdp_state.stations,
            mdp_state.shift_end_time,
            mdp_state.time,
            sorted_station_ids,
        )
        for vid in sorted(mdp_state.vehicles.keys())
    ]
    return torch.tensor(rows, dtype=torch.float32)   # [M, 5]


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

def encode_state(mdp_state: MDPState) -> Dict[str, torch.Tensor]:
    """
    Convert an MDPState into the three tensor blocks consumed by NNValueNetwork.

    This is the single entry point used by NNRolloutPolicy and train_nn_rollout.
    Do not call the individual encode_* helpers directly from outside this module.

    Flow:
        MDPState
            ├─ stations  →  encode_station_block()   →  "station_block"  [N × 5]
            ├─ vehicles  →  encode_vehicle_block()   →  "vehicle_block"  [M × 5]
            └─ (whole)   →  encode_global_context()  →  "global_context"    [7]

    Args:
        mdp_state : MDPState snapshot at the current decision epoch.

    Returns:
        dict with three float32 tensors.

    Downstream usage (NNRolloutPolicy):
        encoded = encode_state(post_decision_state)
        value   = nn_model(
            encoded["station_block"],
            encoded["vehicle_block"],
            encoded["global_context"],
        )
    """
    return {
        "station_block":  encode_station_block(mdp_state),
        "vehicle_block":  encode_vehicle_block(mdp_state),
        "global_context": encode_global_context(mdp_state),
    }
