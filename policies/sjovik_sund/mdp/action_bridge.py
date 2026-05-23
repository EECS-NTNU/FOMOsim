"""
Bridge between canonical MDP actions and simulator Action objects.

This isolates simulator-specific bike-id selection logic from the MDP layer,
so policies can reason in terms of MdpAction and only convert at the boundary
where an action is returned to the simulator.
"""

from __future__ import annotations

from typing import List

from sim.Action import Action

from .mdp_formulation import MdpAction

from settings import MAINTENANCE_REPAIR, MINUTES_PER_ACTION

# Episode-level truncation counters. Reset via reset_truncation_counts() each episode.
truncation_counts: dict = {
    "deliver": 0, "pickup_func": 0, "pickup_depot": 0, "onsite_repair": 0
}
depot_stats: dict = {
    "visits": 0, "loaded_bikes": 0, "unloaded_bikes": 0, "skipped_loads": 0
}

def reset_truncation_counts() -> None:
    for k in truncation_counts:
        truncation_counts[k] = 0
    for k in depot_stats:
        depot_stats[k] = 0

def get_truncation_summary() -> str:
    total = sum(truncation_counts.values())
    trunc = "no truncations" if total == 0 else "truncations: " + ", ".join(f"{k}={v}" for k, v in truncation_counts.items() if v > 0) + f" (total={total})"
    ds = depot_stats
    depot = f"depot: visits={ds['visits']} loaded={ds['loaded_bikes']} unloaded={ds['unloaded_bikes']} skipped_loads={ds['skipped_loads']}"
    return f"{trunc} | {depot}"


def _bike_id(bike) -> int:
    return getattr(bike, "bike_id", getattr(bike, "id"))


def _vehicle_bikes(vehicle) -> list:
    if hasattr(vehicle, "get_bike_inventory"):
        return list(vehicle.get_bike_inventory())

    inventory = getattr(vehicle, "bike_inventory", {})
    if isinstance(inventory, dict):
        return list(inventory.values())
    return list(inventory)


def _station_bikes(station) -> list:
    bikes = getattr(station, "bikes", [])
    if isinstance(bikes, dict):
        return list(bikes.values())
    return list(bikes)


def _take_bike_ids(bikes: list, n: int) -> List[int]:
    if n <= 0:
        return []
    return [_bike_id(b) for b in bikes[:n]]


def record_depot_action_stats(action: Action, vehicle) -> None:
    """Record depot stats for the selected action only."""
    if not getattr(vehicle, "is_at_depot")():
        return

    depot_stats["visits"] += 1

    loaded = len(getattr(action, "pick_ups", []))
    depot_stats["loaded_bikes"] += loaded
    if loaded == 0:
        depot_stats["skipped_loads"] += 1

    vehicle_bikes_by_id = {_bike_id(b): b for b in _vehicle_bikes(vehicle)}
    depot_stats["unloaded_bikes"] += sum(
        1 for b_id in getattr(action, "delivery_bikes", [])
        if getattr(vehicle_bikes_by_id.get(b_id), "damage_status", None) == "depot"
    )


def mdp_action_to_sim_action(
    mdp_action: MdpAction,
    state,
    vehicle,
    maintenance_minutes_per_onsite_repair: float | None = None,
    depot_unload_minutes_per_bike: float = MINUTES_PER_ACTION,
) -> Action:
    """
    Convert a canonical MdpAction into a simulator Action.

    Mapping summary
    - rebalancing > 0 : delivery_bikes (functional bikes from vehicle)
    - rebalancing < 0 : pick_ups (functional bikes from station)
    - depot_removals  : additional pick_ups (depot-damaged bikes from station)
    TODO(doc): The onsite_repairs line below is stale/misleading because
    battery_swaps is hardcoded to [] in the returned simulator Action.
    Clarify that onsite repairs are represented by onsite_repairs plus
    maintenance_time, not battery_swaps.
    - onsite_repairs  : battery_swaps + maintenance_time proxy
    - depot_dropoffs  : delivery_bikes when current station is depot
    - load_from_queue : pick_ups when current station is depot
    """
    if maintenance_minutes_per_onsite_repair is None:
        import settings
        maintenance_minutes_per_onsite_repair = settings.MAINTENANCE_REPAIR

    station = vehicle.location

    station_bikes = _station_bikes(station)
    vehicle_bikes = _vehicle_bikes(vehicle)

    station_functional = [
        b for b in station_bikes if getattr(b, "damage_status", None) not in ("onsite", "depot")
    ]
    station_onsite = [
        b for b in station_bikes if getattr(b, "damage_status", None) == "onsite"
        and not getattr(b, "onsite_repair_in_progress", False)
    ]
    station_depot = [
        b for b in station_bikes if getattr(b, "damage_status", None) == "depot"
    ]

    vehicle_functional = [
        b for b in vehicle_bikes if getattr(b, "damage_status", None) not in ("onsite", "depot")
    ]

    # Track broken bikes in cargo
    vehicle_depot = [
        b for b in vehicle_bikes if getattr(b, "damage_status", None) == "depot"
    ]

    if getattr(vehicle, "is_at_depot")():
        if hasattr(mdp_action, "load_from_queue") and mdp_action.load_from_queue > 0:
            fixed_queue_bikes = list(getattr(station, "fixed_queue", {}).values())
            depot_pickups = _take_bike_ids(fixed_queue_bikes, mdp_action.load_from_queue)
        else:
            depot_pickups = []
        depot_dropoff_count = int(getattr(mdp_action, "depot_dropoffs", 0) or 0)
        depot_dropoffs = _take_bike_ids(vehicle_depot, depot_dropoff_count)
    else:
        depot_pickups = []
        depot_dropoffs = []

    deliver_count = max(int(mdp_action.rebalancing), 0)
    pickup_functional_count = max(-int(mdp_action.rebalancing), 0)
    pickup_depot_count = max(int(mdp_action.depot_removals), 0)
    onsite_repair_count = max(int(mdp_action.onsite_repairs), 0)

    delivery_bikes = _take_bike_ids(vehicle_functional, deliver_count) + depot_dropoffs
    pick_up_functional = _take_bike_ids(station_functional, pickup_functional_count)
    pick_up_depot = _take_bike_ids(station_depot, pickup_depot_count)
    onsite_repairs = _take_bike_ids(station_onsite, onsite_repair_count)

    # Catch silent truncations and accumulate into episode counters.
    if deliver_count > len(vehicle_functional):
        truncation_counts["deliver"] += 1
    if pickup_functional_count > len(station_functional):
        truncation_counts["pickup_func"] += 1
    if pickup_depot_count > len(station_depot):
        truncation_counts["pickup_depot"] += 1
    if onsite_repair_count > len(station_onsite):
        truncation_counts["onsite_repair"] += 1
    if getattr(vehicle, "is_at_depot")() and depot_dropoff_count > len(vehicle_depot):
        print(f"[WARNING - BRIDGE] MDP wanted to drop off {depot_dropoff_count} broken bikes at depot, but vehicle only has {len(vehicle_depot)}. Truncating!")

    # Depot dropoffs are carried in delivery_bikes so Action.get_action_time()
    # already charges MINUTES_PER_ACTION for unloading them.  Keep
    # depot_unload_minutes_per_bike in the signature for backward compatibility,
    # but do not add it to maintenance_time or depot unloads are double-counted.
    return Action(
        battery_swaps=[],
        onsite_repairs=onsite_repairs,
        pick_ups=pick_up_functional + pick_up_depot + depot_pickups,
        delivery_bikes=delivery_bikes,
        next_location=mdp_action.next_station,
        maintenance_time=onsite_repair_count * maintenance_minutes_per_onsite_repair,
    )
