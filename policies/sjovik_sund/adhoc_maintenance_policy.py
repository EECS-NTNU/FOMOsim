"""Ad hoc maintenance-only policy for operational scenario experiments.

This policy uses the maintenance criticality and distance-ranking idea from
``policies/greedy_policy_maintenance.py``, but removes all rebalancing moves.
It can therefore be used as the dedicated maintenance vehicle in separated
operational scenarios without mixing functional-bike rebalancing into the
maintenance routine.
"""

from __future__ import annotations

import sim
from policies import Policy
from settings import MAINTENANCE_REPAIR


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


class AdHocMaintenancePolicy(Policy):
    """Maintenance-only greedy policy with criticality-distance routing.

    The policy never picks up or delivers functional bikes. At each station it:

    1. repairs eligible on-site-fix bikes if ``mode`` allows on-site repair,
    2. collects depot-fix bikes if ``mode`` allows depot collection,
    3. routes either to depot when the broken-bike load threshold is reached,
       or to the station with the best maintenance criticality-distance score.

    Candidate station score:

    ``w_maintenance * eligible_broken_bikes - w_travel * normalized_travel_time``
    """

    def __init__(
        self,
        mode: str = "both",
        tabu_size: int = 1,
        w_maintenance: float = 1.0,
        w_travel: float = 0.1,
        depot_load_threshold: float = 1.0,
        active_window_start_hour: float | None = None,
        active_window_end_hour: float | None = None,
        depot_return_buffer_minutes: float = 10.0,
        idle_wait_minutes: float = 1.0,
        logger=None,
    ):
        super().__init__(maintenance_enabled=True)
        allowed_modes = {"onsite", "depot", "both"}
        if mode not in allowed_modes:
            raise ValueError(f"mode must be one of {sorted(allowed_modes)}, got {mode!r}")

        self.mode = mode
        self.tabu_size = int(tabu_size)
        self.w_maintenance = float(w_maintenance)
        self.w_travel = float(w_travel)
        self.depot_load_threshold = float(depot_load_threshold)
        self.active_window_start_hour = active_window_start_hour
        self.active_window_end_hour = active_window_end_hour
        self.depot_return_buffer_minutes = float(depot_return_buffer_minutes)
        self.idle_wait_minutes = float(idle_wait_minutes)
        self.logger = logger
        self.swap_threshold = 0
        self._tabu: dict[str, list[str]] = {}

    def get_best_action(self, state, vehicle):
        if vehicle.is_at_depot():
            action = self._depot_action(state, vehicle)
        else:
            action = self._station_action(state, vehicle)

        self._update_tabu(vehicle, vehicle.location.id)
        self._log_decision(state, vehicle, action)
        return action

    def _depot_action(self, state, vehicle):
        vehicle_depot = [
            bike
            for bike in _vehicle_bikes(vehicle)
            if getattr(bike, "damage_status", None) == "depot"
        ]
        deliver = [_bike_id(bike) for bike in vehicle_depot]
        depot_service_time = sim.Action([], [], deliver, vehicle.location.id).get_action_time(0.0)

        next_location = self._choose_next_station(
            state,
            vehicle,
            depot_load_after=0,
            service_time=depot_service_time,
        )
        if next_location is None:
            next_location = vehicle.location.id

        maintenance_time = self.idle_wait_minutes if not deliver and next_location == vehicle.location.id else 0.0
        return sim.Action([], [], deliver, next_location, maintenance_time=maintenance_time)

    def _station_action(self, state, vehicle):
        station_bikes = _station_bikes(vehicle.location)
        onsite_ids: list[str] = []
        depot_ids: list[str] = []

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

        depot_load_after = self._depot_load(vehicle) + len(depot_ids)
        maintenance_time = len(onsite_ids) * MAINTENANCE_REPAIR
        service_time = sim.Action(
            [],
            depot_ids,
            [],
            vehicle.location.id,
            maintenance_time=maintenance_time,
            onsite_repairs=onsite_ids,
        ).get_action_time(0.0)
        next_location = self._choose_next_station(
            state,
            vehicle,
            depot_load_after=depot_load_after,
            service_time=service_time,
        )
        if next_location is None:
            next_location = _nearest_depot_id(state, vehicle)

        return sim.Action(
            [],
            depot_ids,
            [],
            next_location,
            maintenance_time=maintenance_time,
            onsite_repairs=onsite_ids,
        )

    def _choose_next_station(self, state, vehicle, depot_load_after: int, service_time: float) -> str | None:
        capacity = max(1, int(getattr(vehicle, "bike_inventory_capacity", 1)))
        nearest_depot = _nearest_depot_id(state, vehicle)
        if depot_load_after >= self.depot_load_threshold * capacity and not vehicle.is_at_depot():
            return nearest_depot

        candidates = []
        tabu = set(self._get_tabu(vehicle))
        for station in state.get_stations():
            if station.id == vehicle.location.id or station.id in tabu:
                continue
            criticality = self._maintenance_criticality(station)
            if criticality <= 0:
                continue
            if not self._can_visit_and_return_to_depot(state, vehicle, station.id, nearest_depot, service_time):
                continue
            candidates.append((station, criticality))

        if not candidates:
            # If tabu blocks all useful stations, try again without tabu.
            for station in state.get_stations():
                if station.id == vehicle.location.id:
                    continue
                criticality = self._maintenance_criticality(station)
                if criticality <= 0:
                    continue
                if not self._can_visit_and_return_to_depot(state, vehicle, station.id, nearest_depot, service_time):
                    continue
                candidates.append((station, criticality))

        if not candidates:
            if self._depot_load(vehicle) > 0 and not vehicle.is_at_depot():
                return nearest_depot
            return None

        travel_times = {
            station.id: state.get_vehicle_travel_time(vehicle.location.id, station.id)
            for station, _ in candidates
        }
        max_travel = max(travel_times.values()) or 1.0
        scores = {}
        for station, criticality in candidates:
            travel_penalty = travel_times[station.id] / max_travel
            scores[station.id] = self.w_maintenance * criticality - self.w_travel * travel_penalty
        return max(scores, key=scores.__getitem__)

    def _maintenance_criticality(self, station) -> int:
        onsite_count = 0
        depot_count = 0
        for bike in _station_bikes(station):
            status = getattr(bike, "damage_status", None)
            if self.mode in {"onsite", "both"} and status == "onsite":
                if not getattr(bike, "onsite_repair_in_progress", False):
                    onsite_count += 1
            if self.mode in {"depot", "both"} and status == "depot":
                depot_count += 1
        return onsite_count + depot_count

    def _can_visit_and_return_to_depot(
        self,
        state,
        vehicle,
        station_id: str,
        depot_id: str,
        service_time: float,
    ) -> bool:
        if self.active_window_start_hour is None or self.active_window_end_hour is None:
            return True
        start = float(self.active_window_start_hour) % 24.0
        end = float(self.active_window_end_hour) % 24.0
        if start == end:
            return True

        clock_min = float(state.time % 1440.0)
        end_min = end * 60.0
        if start > end and clock_min >= start * 60.0:
            end_min += 1440.0
        if end_min < clock_min:
            end_min += 1440.0

        time_remaining = max(0.0, end_min - clock_min)
        travel_to_station = state.get_vehicle_travel_time(vehicle.location.id, station_id)
        travel_to_depot = state.get_vehicle_travel_time(station_id, depot_id)
        required_time = (
            service_time
            + travel_to_station
            + travel_to_depot
            + self.depot_return_buffer_minutes
        )
        return required_time <= time_remaining

    def _depot_load(self, vehicle) -> int:
        return sum(
            1
            for bike in _vehicle_bikes(vehicle)
            if getattr(bike, "damage_status", None) == "depot"
        )

    def _get_tabu(self, vehicle) -> list[str]:
        return self._tabu.get(vehicle.id, [])

    def _update_tabu(self, vehicle, station_id: str) -> None:
        history = self._tabu.get(vehicle.id, [])
        history.append(station_id)
        self._tabu[vehicle.id] = history[-self.tabu_size :]

    def _log_decision(self, state, vehicle, action) -> None:
        logger = self.logger
        if logger is None:
            return

        current_time = state.time
        clock_min = current_time % (24 * 60)
        row = {
            "day": int(current_time // (24 * 60)),
            "hour": int(clock_min // 60),
            "minute": int(clock_min % 60),
            **self._action_counts(state, vehicle, action),
            "immediate_reward": "",
            "vfa_value": "",
            "accumulated_rollout_reward": "",
            "tail_value": "",
            "final_decision_score": "",
            "n_total_candidates": "",
            "vfa_top1_next_station": "",
            "rollout_changed_decision": "",
            "winning_candidate_rank": "",
            "decision_runtime_s": "",
            "winning_profile_type": f"adhoc-maintenance-{self.mode}",
        }
        logger.log_decision(row)

    def _action_counts(self, state, vehicle, action) -> dict:
        inv = _vehicle_bikes(vehicle)
        func_before = sum(
            1 for bike in inv if getattr(bike, "damage_status", None) not in ("depot", "onsite")
        )
        depot_before = self._depot_load(vehicle)
        total_before = len(inv)
        station_bikes = {_bike_id(bike): bike for bike in _station_bikes(vehicle.location)}
        vehicle_bikes = {_bike_id(bike): bike for bike in inv}
        is_at_depot = vehicle.is_at_depot()

        pickup_ids = list(getattr(action, "pick_ups", []) or [])
        delivery_ids = list(getattr(action, "delivery_bikes", []) or [])
        onsite_repairs = len(getattr(action, "onsite_repairs", []) or [])

        depot_pickups = 0
        if not is_at_depot:
            depot_pickups = sum(
                1
                for bike_id in pickup_ids
                if getattr(station_bikes.get(bike_id), "damage_status", None) == "depot"
            )
        depot_deliveries = 0
        if is_at_depot:
            depot_deliveries = sum(
                1
                for bike_id in delivery_ids
                if getattr(vehicle_bikes.get(bike_id), "damage_status", None) == "depot"
            )

        depot_after = max(depot_before - depot_deliveries, 0) + depot_pickups
        total_after = max(func_before, 0) + max(depot_after, 0)

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
            1
            for bike in station_bikes.values()
            if getattr(bike, "damage_status", None) is not None
        )
        is_maintenance = onsite_repairs > 0 or depot_pickups > 0 or depot_deliveries > 0

        return {
            "current_station_id": vehicle.location.id,
            "is_at_depot": is_at_depot,
            "functional_load_before": func_before,
            "depot_load_before": depot_before,
            "total_load_before": total_before,
            "functional_deliveries": 0,
            "functional_pickups": 0,
            "onsite_repairs": onsite_repairs,
            "depot_pickups": depot_pickups,
            "depot_deliveries": depot_deliveries,
            "load_from_queue": 0,
            "bikes_involved": str(pickup_ids + delivery_ids + list(getattr(action, "onsite_repairs", []) or [])),
            "action_duration_min": round(action_duration, 3),
            "next_station_id": str(dest) if dest else "",
            "travel_time_min": round(travel_time, 3),
            "functional_load_after": func_before,
            "depot_load_after": max(depot_after, 0),
            "total_load_after": total_after,
            "maintenance_flag_present": n_damaged > 0 or depot_before > 0 or is_at_depot,
            "selected_action_is_maintenance": is_maintenance,
        }

    def __repr__(self) -> str:
        return f"AdHocMaintenancePolicy(mode={self.mode})"
