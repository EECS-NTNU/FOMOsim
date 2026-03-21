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


def mdp_action_to_sim_action(
    mdp_action: MdpAction,
    state,
    vehicle,
    maintenance_minutes_per_onsite_repair: float = 5.0,
) -> Action:
    """
    Convert a canonical MdpAction into a simulator Action.

    Mapping summary
    - rebalancing > 0 : delivery_bikes (functional bikes from vehicle)
    - rebalancing < 0 : pick_ups (functional bikes from station)
    - depot_removals  : additional pick_ups (depot-damaged bikes from station)
    - onsite_repairs  : battery_swaps + maintenance_time proxy
    - load_from_queue : pick_ups when current station is depot
    """
    station = vehicle.location

    station_bikes = _station_bikes(station)
    vehicle_bikes = _vehicle_bikes(vehicle)

    station_functional = [
        b for b in station_bikes if getattr(b, "damage_status", None) not in ("onsite", "depot")
    ]
    station_onsite = [
        b for b in station_bikes if getattr(b, "damage_status", None) == "onsite"
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
        # NEW: Automatically unload ALL broken bikes at depot
        depot_dropoffs = _take_bike_ids(vehicle_depot, len(vehicle_depot))
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

    # --- NEW: Catch Silent Truncations ---
    if deliver_count > len(vehicle_functional):
        print(f"[WARNING - BRIDGE] MDP wanted to deliver {deliver_count} bikes, but vehicle only has {len(vehicle_functional)} functional. Truncating!")
    if pickup_functional_count > len(station_functional):
        print(f"[WARNING - BRIDGE] MDP wanted to pick up {pickup_functional_count} functional bikes, but station only has {len(station_functional)}. Truncating!")
    if pickup_depot_count > len(station_depot):
        print(f"[WARNING - BRIDGE] MDP wanted to pick up {pickup_depot_count} broken bikes, but station only has {len(station_depot)}. Truncating!")
    if onsite_repair_count > len(station_onsite):
        print(f"[WARNING - BRIDGE] MDP wanted to repair {onsite_repair_count} bikes onsite, but station only has {len(station_onsite)}. Truncating!")

    return Action(
        battery_swaps=[], # Add empty list as we do not consider battery swaps in this policy
        onsite_repairs=onsite_repairs,
        pick_ups=pick_up_functional + pick_up_depot + depot_pickups,
        delivery_bikes=delivery_bikes,
        next_location=mdp_action.next_station,
        maintenance_time=onsite_repair_count * maintenance_minutes_per_onsite_repair,
    )
