"""
candidate_generator.py  —  Action Candidate Generation for DSJBRMP

Standalone function that enumerates tractable (MdpAction, sim.Action) pairs
for a vehicle arrival decision. Previously lived as LinearVFAPolicy._generate_candidates;
moved here so both the VFA and the NN branches can import it without
either policy depending on the other.

─────────────────────────────────────────────────────────────────────────────
ACTION-SPACE SPLITTING
─────────────────────────────────────────────────────────────────────────────

The full action space (routing × inventory × maintenance) is intractable to
enumerate. We reduce it by splitting into two levels:

  Micro (inventory)  — greedy; fixed given the routing choice.
    • Rebalancing: push the current station toward its target inventory.
    • Maintenance:  pick up all depot-damaged bikes that fit; enumerate
                    0…all possible on-site repairs (one candidate per count).

  Macro (routing)   — enumerated; this is what varies across candidates.
    • Pool = 5 nearest stations (by travel time)
             + 3 most critical stations (highest |current − target| deviation)
    • Depot is always included (not subject to tabu).
    • Tabu list: stations already claimed by en-route vehicles are excluded
                 to prevent two vehicles racing to the same starving station.

─────────────────────────────────────────────────────────────────────────────
RETURN FORMAT
─────────────────────────────────────────────────────────────────────────────

  return_pairs=False (default):  List[sim.Action]
      Backward-compatible; used by LinearVFAPolicy and HybridRolloutPolicy.

  return_pairs=True:             List[Tuple[MdpAction, sim.Action]]
      Used by NNLearningPolicy and NNRolloutPolicy, which need the raw
      MdpAction to compute PostDecisionState.apply() for NN value encoding.
"""

from typing import List, Tuple, Union

from policies.sjovik_sund.mdp.mdp_formulation import MdpAction
from policies.sjovik_sund.mdp.action_bridge import mdp_action_to_sim_action


# ── Routing pool constants ────────────────────────────────────────────────────
# These control how large the candidate set can be.
# Increasing N_NEAREST / N_CRITICAL improves coverage but slows rollout.
N_NEAREST   = 5   # closest stations by travel time
N_CRITICAL  = 3   # most off-target stations by |inventory − target|
N_FALLBACK  = 8   # pool cap when tabu list removes all primary candidates


def generate_candidates(
    state,
    vehicle,
    maintenance_enabled: bool,
    n_fallback: int = N_FALLBACK,
    verbose: bool = False,
    return_pairs: bool = False,
) -> Union[List, List[Tuple]]:
    """
    Generate a tractable set of candidate actions for the arriving vehicle.

    Args:
        state               : sim.State — live simulator state.
        vehicle             : sim.Vehicle — the vehicle making the decision.
        maintenance_enabled : whether depot removals and on-site repairs
                              are included as action dimensions.
        n_fallback          : pool size cap used only when the tabu list
                              removes all primary candidates (default 8).
        verbose             : if True, print tabu list debug info.
        return_pairs        : if True, return List[(MdpAction, sim.Action)];
                              if False, return List[sim.Action] (default).

    Returns:
        List of sim.Action objects, or (MdpAction, sim.Action) pairs
        depending on return_pairs.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # MICRO LEVEL: compute inventory quantities for the current station
    # ─────────────────────────────────────────────────────────────────────────

    load_from_queue = 0  # only used at the depot

    if vehicle.is_at_depot():
        # At the depot: no rebalancing or repairs; pick up all repaired bikes
        # that fit in the vehicle's remaining capacity.
        rebalancing    = 0
        depot_removals = 0
        n_vehicle      = len(vehicle.get_bike_inventory())
        vehicle_capacity = int(
            getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle))
        )
        free_cap            = max(0, vehicle_capacity - n_vehicle)
        repaired_available  = len(getattr(vehicle.location, "fixed_queue", {}))
        load_from_queue     = min(repaired_available, free_cap)
        onsite_repairs_options = [0]   # on-site repairs not possible at depot

    else:
        # At a normal station: compute target, inventories, rebalancing delta.
        target = round(vehicle.location.get_target_state(state.day(), state.hour()))

        # Functional bikes only — broken bikes do not count toward service level
        functional_bikes = [
            b for b in vehicle.location.get_bikes() if getattr(b, "is_available", True)
        ]
        n_station = len(functional_bikes)

        # Vehicle inventory: split into functional vs. broken
        inv             = vehicle.get_bike_inventory()
        n_vehicle_total = len(inv)
        n_vehicle_func  = sum(
            1 for b in inv if getattr(b, "damage_status", None) not in ["depot", "onsite"]
        )
        vehicle_capacity = int(
            getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle_total))
        )
        # Free capacity must use TOTAL bikes on board — broken bikes occupy physical space.
        free_cap = max(0, vehicle_capacity - n_vehicle_total)

        # Greedily pick up all depot-damaged bikes at this station (if maintenance enabled)
        depot_removals = 0
        onsite_bikes   = []
        if maintenance_enabled:
            broken_bikes   = [
                b for b in vehicle.location.bikes.values()
                if getattr(b, "damage_status", None) == "depot"
            ]
            depot_removals = min(len(broken_bikes), free_cap)
            onsite_bikes   = [
                b for b in vehicle.location.bikes.values()
                if getattr(b, "damage_status", None) == "onsite"
            ]

        # On-site repair enumeration: one candidate per repair count (0 to all).
        # Each count yields a different post-repair inventory and thus a different
        # rebalancing delta, so we generate a separate action for each.
        num_onsite             = len(onsite_bikes)
        onsite_repairs_options = list(range(0, num_onsite + 1))

        # Base rebalancing (used as starting point; adjusted per onsite_repairs below)
        delta = target - n_station   # >0 → deliver, <0 → pick up
        if delta > 0:
            rebalancing = min(n_vehicle_func, delta)
        elif delta < 0:
            rebalancing = -min(n_station, -delta, max(free_cap - depot_removals, 0))
        else:
            rebalancing = 0

    # ─────────────────────────────────────────────────────────────────────────
    # TABU LIST: exclude stations already claimed by en-route vehicles
    # ─────────────────────────────────────────────────────────────────────────
    # Simple multi-vehicle coordination: if another vehicle is already heading
    # to a station (eta in the future), we skip that station. This prevents
    # two vehicles converging on the same starved location simultaneously.

    claimed_stations = set()
    for v in state.get_vehicles():
        if v.id != vehicle.id:
            v_eta = getattr(v, "eta", 0)
            if v_eta > state.time:
                dest = (
                    getattr(v, "destination_station", None)
                    or getattr(v, "next_location", None)
                    or (v.location.id if v.location else None)
                )
                if dest:
                    claimed_stations.add(dest)

    if verbose:
        print(f"[TABU] Vehicle {vehicle.id} | Claimed stations: {claimed_stations}")

    # ─────────────────────────────────────────────────────────────────────────
    # MACRO LEVEL: build routing candidate pool
    # ─────────────────────────────────────────────────────────────────────────

    cur_id = vehicle.location.id

    # Start from all non-current, non-claimed stations
    pool = [
        s for s in state.get_stations()
        if s.id != cur_id and s.id not in claimed_stations
    ]

    # Depot is never subject to tabu — repairs are a global resource
    depot_stations = state.get_depots()
    if depot_stations:
        depot = depot_stations[0]
        if depot.id != cur_id:
            pool.append(depot)

    # Myopic blindspot reduction: 5 nearest + 3 most critical.
    # Using ONLY nearest-N would miss urgent distant stations.
    # Using ONLY critical would ignore fast wins nearby.
    # The combination gives reasonable spatial and urgency coverage.
    pool.sort(key=lambda s: state.get_travel_time(cur_id, s.id))
    nearest_stations  = pool[:N_NEAREST]
    remaining_pool    = pool[N_NEAREST:]
    remaining_pool.sort(
        key=lambda s: abs(
            s.get_target_state(state.day(), state.hour()) - len(s.get_bikes())
        ),
        reverse=True,
    )
    critical_stations = remaining_pool[:N_CRITICAL]
    pool = nearest_stations + critical_stations

    # Fallback: if tabu removed all candidates, open up to any reachable station
    if not pool:
        pool = [s for s in state.get_stations() if s.id != cur_id]
        if depot_stations:
            depot = depot_stations[0]
            if depot.id != cur_id:
                pool.append(depot)
        pool.sort(key=lambda s: state.get_travel_time(cur_id, s.id))
        pool = pool[:n_fallback]

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD (MdpAction, sim.Action) PAIRS
    # ─────────────────────────────────────────────────────────────────────────
    # One pair per (station, onsite_repair_count) combination.
    # The rebalancing quantity is re-computed for each repair count because
    # repairing bikes on-site changes the effective station inventory, which
    # in turn changes how many functional bikes need to be delivered or picked up.

    pairs = []
    for s in pool:
        for onsite_repairs in onsite_repairs_options:

            if vehicle.is_at_depot():
                current_rebalancing = 0
            else:
                # Recompute rebalancing delta after accounting for on-site repairs
                new_n_station = n_station + onsite_repairs
                delta         = target - new_n_station
                if delta > 0:
                    current_rebalancing = min(n_vehicle_func, delta)
                elif delta < 0:
                    current_rebalancing = -min(
                        new_n_station, -delta, max(free_cap - depot_removals, 0)
                    )
                else:
                    current_rebalancing = 0

            mdp_action = MdpAction(
                current_station=cur_id,
                rebalancing=int(current_rebalancing),
                onsite_repairs=int(onsite_repairs),
                depot_removals=int(depot_removals),
                load_from_queue=int(load_from_queue),
                next_station=s.id,
            )
            pairs.append((mdp_action, mdp_action_to_sim_action(mdp_action, state, vehicle)))

    # Final fallback: if the pool was completely empty, route to the closest depot
    if not pairs:
        depot_id = state.get_closest_depot(vehicle)
        for onsite_repairs in onsite_repairs_options:
            if vehicle.is_at_depot():
                current_rebalancing = 0
            else:
                new_n_station = n_station + onsite_repairs
                delta         = target - new_n_station
                if delta > 0:
                    current_rebalancing = min(n_vehicle_func, delta)
                elif delta < 0:
                    current_rebalancing = -min(
                        new_n_station, -delta, max(free_cap - depot_removals, 0)
                    )
                else:
                    current_rebalancing = 0

            mdp_action = MdpAction(
                current_station=cur_id,
                rebalancing=int(current_rebalancing),
                onsite_repairs=int(onsite_repairs),
                depot_removals=int(depot_removals),
                load_from_queue=int(load_from_queue),
                next_station=depot_id,
            )
            pairs.append((mdp_action, mdp_action_to_sim_action(mdp_action, state, vehicle)))

    # ─────────────────────────────────────────────────────────────────────────
    # RETURN
    # ─────────────────────────────────────────────────────────────────────────

    if return_pairs:
        # Full pairs: needed by NNLearningPolicy and NNRolloutPolicy so they can
        # call PostDecisionState.apply(mdp_action) to encode S^x for the NN.
        return pairs

    # Flat sim.Action list: backward-compatible default used by the VFA and
    # HybridRolloutPolicy, which only need sim.Action objects for execution.
    return [sim_action for _, sim_action in pairs]
