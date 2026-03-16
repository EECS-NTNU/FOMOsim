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

    if getattr(vehicle, "is_at_depot")() and mdp_action.load_from_queue > 0:
        depot_pickups = _take_bike_ids(station_functional, mdp_action.load_from_queue)
    else:
        depot_pickups = []

    deliver_count = max(int(mdp_action.rebalancing), 0)
    pickup_functional_count = max(-int(mdp_action.rebalancing), 0)
    pickup_depot_count = max(int(mdp_action.depot_removals), 0)
    onsite_repair_count = max(int(mdp_action.onsite_repairs), 0)

    delivery_bikes = _take_bike_ids(vehicle_functional, deliver_count)
    pick_up_functional = _take_bike_ids(station_functional, pickup_functional_count)
    pick_up_depot = _take_bike_ids(station_depot, pickup_depot_count)
    battery_swaps = _take_bike_ids(station_onsite, onsite_repair_count)

    return Action(
        battery_swaps=battery_swaps,
        pick_ups=pick_up_functional + pick_up_depot + depot_pickups,
        delivery_bikes=delivery_bikes,
        next_location=mdp_action.next_station,
        maintenance_time=onsite_repair_count * maintenance_minutes_per_onsite_repair,
    )
