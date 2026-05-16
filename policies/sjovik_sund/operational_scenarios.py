"""Operational scenario policies for separated rebalancing and maintenance.

The simulator attaches one policy object to each vehicle, but that policy is
called with the concrete vehicle at every decision epoch.  This module uses
that hook to dispatch vehicles to different operating rules while keeping the
existing VFA and Hybrid policies unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sim
from policies import Policy
from settings import MAINTENANCE_REPAIR, SERVICE_TIME_FROM, SERVICE_TIME_TO


@dataclass(frozen=True)
class TimeWindow:
    """Hour-of-day activation window.

    start_hour and end_hour use 24-hour clock values.  Windows may wrap
    midnight, e.g. 20-6.  Equal start and end means active all day.
    """

    start_hour: float
    end_hour: float

    @classmethod
    def parse(cls, text: str | None) -> "TimeWindow":
        if text is None or str(text).strip().lower() in {"", "all", "always"}:
            return cls(0.0, 0.0)
        cleaned = str(text).strip().replace(":", ".")
        if "-" not in cleaned:
            hour = float(cleaned)
            return cls(hour, hour)
        start, end = cleaned.split("-", 1)
        return cls(float(start), float(end))

    def contains(self, hour: float) -> bool:
        hour = float(hour) % 24.0
        start = self.start_hour % 24.0
        end = self.end_hour % 24.0
        if start == end:
            return True
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    def label(self) -> str:
        if self.start_hour % 24.0 == self.end_hour % 24.0:
            return "all_day"
        return f"{self.start_hour:g}-{self.end_hour:g}"


@dataclass(frozen=True)
class VehiclePolicyAssignment:
    vehicle_id: str
    policy: Policy
    role: str
    active_window: TimeWindow
    idle_outside_window: bool = True


def _bike_id(bike) -> str:
    return getattr(bike, "bike_id", getattr(bike, "id", str(bike)))


def _station_bikes(location) -> list:
    bikes = getattr(location, "bikes", {})
    if isinstance(bikes, dict):
        return list(bikes.values())
    return list(bikes)


def _vehicle_bikes(vehicle) -> list:
    if hasattr(vehicle, "get_bike_inventory"):
        return list(vehicle.get_bike_inventory())
    inventory = getattr(vehicle, "bike_inventory", {})
    if isinstance(inventory, dict):
        return list(inventory.values())
    return list(inventory)


def _nearest_depot_id(state, vehicle) -> str:
    depots = list(state.get_depots())
    if not depots:
        return vehicle.location.id
    return min(
        depots,
        key=lambda depot: state.get_vehicle_travel_time(vehicle.location.id, depot.id),
    ).id


def _action_counts(state, vehicle, action) -> dict:
    """Return decision-log fields for an arbitrary simulator action."""
    inv = _vehicle_bikes(vehicle)
    func_before = sum(
        1 for b in inv if getattr(b, "damage_status", None) not in ("depot", "onsite")
    )
    depot_before = sum(1 for b in inv if getattr(b, "damage_status", None) == "depot")
    total_before = len(inv)

    station_bikes = {_bike_id(b): b for b in _station_bikes(vehicle.location)}
    vehicle_bikes = {_bike_id(b): b for b in inv}
    is_at_depot = vehicle.is_at_depot()
    fixed_queue = getattr(vehicle.location, "fixed_queue", {})

    pick_up_ids = list(getattr(action, "pick_ups", []) or [])
    delivery_ids = list(getattr(action, "delivery_bikes", []) or [])
    onsite_repairs = len(getattr(action, "onsite_repairs", []) or [])

    func_pickups = 0
    depot_pickups = 0
    load_from_queue = 0
    if is_at_depot:
        load_from_queue = sum(1 for bike_id in pick_up_ids if bike_id in fixed_queue)
    else:
        for bike_id in pick_up_ids:
            bike = station_bikes.get(bike_id)
            if bike and getattr(bike, "damage_status", None) == "depot":
                depot_pickups += 1
            elif bike:
                func_pickups += 1

    if is_at_depot:
        depot_deliveries = sum(
            1
            for bike_id in delivery_ids
            if getattr(vehicle_bikes.get(bike_id), "damage_status", None) == "depot"
        )
        func_deliveries = len(delivery_ids) - depot_deliveries
    else:
        depot_deliveries = 0
        func_deliveries = len(delivery_ids)

    if is_at_depot:
        func_after = func_before - func_deliveries + load_from_queue
        depot_after = max(depot_before - depot_deliveries, 0)
    else:
        func_after = func_before - func_deliveries + func_pickups
        depot_after = depot_before + depot_pickups
    total_after = max(func_after, 0) + max(depot_after, 0)

    dest = getattr(action, "next_location", None)
    try:
        travel_time = state.get_vehicle_travel_time(vehicle.location.id, dest) if dest else 0.0
    except Exception:
        travel_time = 0.0
    try:
        action_duration = action.get_action_time(0.0) if hasattr(action, "get_action_time") else 0.0
    except Exception:
        action_duration = 0.0

    n_damaged = sum(
        1 for bike in station_bikes.values() if getattr(bike, "damage_status", None) is not None
    )
    is_maintenance = (
        onsite_repairs > 0
        or depot_pickups > 0
        or depot_deliveries > 0
        or load_from_queue > 0
        or is_at_depot
    )

    return {
        "current_station_id": vehicle.location.id,
        "is_at_depot": is_at_depot,
        "functional_load_before": func_before,
        "depot_load_before": depot_before,
        "total_load_before": total_before,
        "functional_deliveries": func_deliveries,
        "functional_pickups": func_pickups,
        "onsite_repairs": onsite_repairs,
        "depot_pickups": depot_pickups,
        "depot_deliveries": depot_deliveries,
        "load_from_queue": load_from_queue,
        "bikes_involved": str(
            pick_up_ids + delivery_ids + list(getattr(action, "onsite_repairs", []) or [])
        ),
        "action_duration_min": round(action_duration, 3),
        "next_station_id": str(dest) if dest else "",
        "travel_time_min": round(travel_time, 3),
        "functional_load_after": max(func_after, 0),
        "depot_load_after": max(depot_after, 0),
        "total_load_after": total_after,
        "maintenance_flag_present": (n_damaged > 0) or is_at_depot,
        "selected_action_is_maintenance": is_maintenance,
    }


def log_policy_action(
    logger,
    state,
    vehicle,
    action,
    *,
    winning_profile_type: str,
    decision_runtime_s: float | str = "",
) -> None:
    """Write a generic decision row for non-VFA policies."""
    if logger is None:
        return

    current_time = state.time
    clock_min = current_time % (24 * 60)
    row = {
        "day": int(current_time // (24 * 60)),
        "hour": int(clock_min // 60),
        "minute": int(clock_min % 60),
        **_action_counts(state, vehicle, action),
        "immediate_reward": "",
        "vfa_value": "",
        "accumulated_rollout_reward": "",
        "tail_value": "",
        "final_decision_score": "",
        "n_total_candidates": "",
        "vfa_top1_next_station": "",
        "rollout_changed_decision": "",
        "winning_candidate_rank": "",
        "decision_runtime_s": decision_runtime_s,
        "winning_profile_type": winning_profile_type,
    }
    logger.log_decision(row)


class MaintenanceTourPolicy(Policy):
    """Shortest-path style maintenance-only policy.

    The policy never moves functional bikes for rebalancing.  It repairs
    on-site-fix bikes and/or collects depot-fix bikes according to mode, then
    routes to the nearest remaining station with eligible broken bikes.
    """

    def __init__(
        self,
        mode: str = "both",
        depot_load_threshold: float = 1.0,
        load_repaired_from_depot: bool = False,
        logger=None,
    ):
        super().__init__(maintenance_enabled=True)
        allowed_modes = {"onsite", "depot", "both"}
        if mode not in allowed_modes:
            raise ValueError(f"mode must be one of {sorted(allowed_modes)}, got {mode!r}")
        self.mode = mode
        self.depot_load_threshold = float(depot_load_threshold)
        self.load_repaired_from_depot = bool(load_repaired_from_depot)
        self.logger = logger
        self.swap_threshold = 0

    def get_best_action(self, state, vehicle):
        if vehicle.is_at_depot():
            action = self._depot_action(state, vehicle)
        else:
            action = self._station_action(state, vehicle)
        log_policy_action(
            self.logger,
            state,
            vehicle,
            action,
            winning_profile_type=f"maintenance-tour-{self.mode}",
        )
        return action

    def _depot_action(self, state, vehicle):
        vehicle_depot = [
            bike for bike in _vehicle_bikes(vehicle) if getattr(bike, "damage_status", None) == "depot"
        ]
        deliver = [_bike_id(bike) for bike in vehicle_depot]

        pickup = []
        if self.load_repaired_from_depot:
            remaining_capacity = max(
                0,
                vehicle.bike_inventory_capacity - (len(_vehicle_bikes(vehicle)) - len(deliver)),
            )
            fixed_queue = list(getattr(vehicle.location, "fixed_queue", {}).keys())
            pickup = fixed_queue[:remaining_capacity]

        next_location = self._nearest_maintenance_station(state, vehicle)
        if next_location is None:
            next_location = vehicle.location.id
        return sim.Action([], pickup, deliver, next_location, maintenance_time=0.0)

    def _station_action(self, state, vehicle):
        station_bikes = _station_bikes(vehicle.location)
        onsite_ids = []
        depot_ids = []

        if self.mode in {"onsite", "both"}:
            onsite_ids = [
                _bike_id(bike)
                for bike in station_bikes
                if getattr(bike, "damage_status", None) == "onsite"
                and not getattr(bike, "onsite_repair_in_progress", False)
            ]

        if self.mode in {"depot", "both"}:
            free_capacity = max(0, vehicle.bike_inventory_capacity - len(_vehicle_bikes(vehicle)))
            depot_ids = [
                _bike_id(bike)
                for bike in station_bikes
                if getattr(bike, "damage_status", None) == "depot"
            ][:free_capacity]

        depot_load_after = sum(
            1 for bike in _vehicle_bikes(vehicle) if getattr(bike, "damage_status", None) == "depot"
        ) + len(depot_ids)
        capacity = max(1, vehicle.bike_inventory_capacity)
        if depot_load_after >= self.depot_load_threshold * capacity:
            next_location = _nearest_depot_id(state, vehicle)
        else:
            next_location = self._nearest_maintenance_station(state, vehicle)
            if next_location is None:
                next_location = _nearest_depot_id(state, vehicle)

        maintenance_time = len(onsite_ids) * MAINTENANCE_REPAIR
        return sim.Action(
            [],
            depot_ids,
            [],
            next_location,
            maintenance_time=maintenance_time,
            onsite_repairs=onsite_ids,
        )

    def _nearest_maintenance_station(self, state, vehicle) -> str | None:
        candidates = []
        for station in state.get_stations():
            if station.id == vehicle.location.id:
                continue
            score = self._eligible_broken_count(station)
            if score <= 0:
                continue
            candidates.append(station)

        if not candidates:
            return None
        return min(
            candidates,
            key=lambda station: state.get_vehicle_travel_time(vehicle.location.id, station.id),
        ).id

    def _eligible_broken_count(self, station) -> int:
        count = 0
        for bike in _station_bikes(station):
            status = getattr(bike, "damage_status", None)
            if self.mode in {"onsite", "both"} and status == "onsite":
                if not getattr(bike, "onsite_repair_in_progress", False):
                    count += 1
            if self.mode in {"depot", "both"} and status == "depot":
                count += 1
        return count

    def __repr__(self) -> str:
        return f"MaintenanceTourPolicy(mode={self.mode})"


class VehiclePolicyDispatcher(Policy):
    """Dispatch vehicles to role-specific policies and activation windows."""

    def __init__(
        self,
        default_policy: Policy,
        assignments: Iterable[VehiclePolicyAssignment] | None = None,
        default_window: TimeWindow | None = None,
        logger=None,
        scenario_name: str = "",
    ):
        super().__init__(maintenance_enabled=getattr(default_policy, "maintenance_enabled", True))
        self.default_policy = default_policy
        self.assignments = {assignment.vehicle_id: assignment for assignment in assignments or []}
        self.default_window = default_window or TimeWindow(float(SERVICE_TIME_FROM), float(SERVICE_TIME_TO))
        self.logger = logger
        self.scenario_name = scenario_name
        self.swap_threshold = getattr(default_policy, "swap_threshold", 0)

    def init_sim(self, simulator) -> None:
        seen = set()
        for policy in [self.default_policy] + [assignment.policy for assignment in self.assignments.values()]:
            if id(policy) in seen:
                continue
            seen.add(id(policy))
            if hasattr(policy, "init_sim"):
                policy.init_sim(simulator)

    def get_action(self, state, vehicle):
        hour = state.hour()
        assignment = self.assignments.get(vehicle.id)
        if assignment is not None:
            active = assignment.active_window.contains(hour)
            if active:
                return self._delegate(
                    state,
                    vehicle,
                    assignment.policy,
                    vehicle_role=assignment.role,
                    active_window=assignment.active_window.label(),
                    vehicle_active=True,
                )
            if assignment.idle_outside_window:
                action = self._return_to_depot_action(state, vehicle)
                self._with_context(
                    vehicle,
                    vehicle_role=assignment.role,
                    vehicle_policy_type="inactive",
                    active_window=assignment.active_window.label(),
                    vehicle_active=False,
                )
                try:
                    log_policy_action(
                        self.logger,
                        state,
                        vehicle,
                        action,
                        winning_profile_type="inactive-return-to-depot",
                    )
                finally:
                    self._clear_context()
                return action

        if self.default_window.contains(hour):
            return self._delegate(
                state,
                vehicle,
                self.default_policy,
                vehicle_role="default",
                active_window=self.default_window.label(),
                vehicle_active=True,
            )

        action = self._return_to_depot_action(state, vehicle)
        self._with_context(
            vehicle,
            vehicle_role="default",
            vehicle_policy_type="inactive",
            active_window=self.default_window.label(),
            vehicle_active=False,
        )
        try:
            log_policy_action(
                self.logger,
                state,
                vehicle,
                action,
                winning_profile_type="inactive-return-to-depot",
            )
        finally:
            self._clear_context()
        return action

    def get_best_action(self, state, vehicle):
        return self.get_action(state, vehicle)

    def _delegate(
        self,
        state,
        vehicle,
        policy,
        *,
        vehicle_role: str,
        active_window: str,
        vehicle_active: bool,
    ):
        self._with_context(
            vehicle,
            vehicle_role=vehicle_role,
            vehicle_policy_type=policy.__class__.__name__,
            active_window=active_window,
            vehicle_active=vehicle_active,
        )
        try:
            return policy.get_best_action(state, vehicle)
        finally:
            self._clear_context()

    def _with_context(
        self,
        vehicle,
        *,
        vehicle_role: str,
        vehicle_policy_type: str,
        active_window: str,
        vehicle_active: bool,
    ) -> None:
        if self.logger is None or not hasattr(self.logger, "set_decision_context"):
            return
        self.logger.set_decision_context(
            {
                "scenario_name": self.scenario_name,
                "vehicle_id": vehicle.id,
                "vehicle_role": vehicle_role,
                "vehicle_policy_type": vehicle_policy_type,
                "active_time_window": active_window,
                "vehicle_active": vehicle_active,
            }
        )

    def _clear_context(self) -> None:
        if self.logger is not None and hasattr(self.logger, "clear_decision_context"):
            self.logger.clear_decision_context()

    def _return_to_depot_action(self, state, vehicle):
        return sim.Action([], [], [], _nearest_depot_id(state, vehicle), maintenance_time=0.0)

    def __repr__(self) -> str:
        return f"VehiclePolicyDispatcher({self.scenario_name or 'scenario'})"
