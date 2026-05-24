"""
Greedy maintenance-aware baseline policy.
 
Priority order at each visit:
  1. On-site repairs (all fixable bikes at current station)
  2. Depot bike pick-up / drop-off (broken bikes in, repaired bikes out)
  3. Rebalancing toward target state
  4. Battery swaps (when delivering)
 
Routing: score every candidate station on maintenance urgency + rebalancing need,
penalised by travel time. Tabu keeps the vehicle from immediately backtracking.
"""
 
from policies import Policy
import sim
from settings import *
 
 
class GreedyMaintenancePolicy(Policy):
    """
    Strong greedy baseline that aggressively handles maintenance
    while still rebalancing toward target state.
    """
 
    def __init__(
        self,
        tabu_size: int = 1,
        w_maintenance: float = 1.0,
        w_rebalancing: float = 1.0,
        w_travel: float = 0.1,
        depot_load_threshold: float = 0.5,
        depot_return_buffer_minutes: float = 10.0,
        swap_threshold=BATTERY_LIMIT_TO_SWAP,
    ):
        """
        w_maintenance   : weight for stations with broken/onsite bikes
        w_rebalancing   : weight for deviation from target state
        w_travel        : weight penalising far-away stations
        depot_load_threshold : fraction of vehicle capacity that, when filled
                               with depot bikes, triggers a depot trip
        """
        self.tabu_size = tabu_size
        self.w_maintenance = w_maintenance
        self.w_rebalancing = w_rebalancing
        self.w_travel = w_travel
        self.depot_load_threshold = depot_load_threshold
        self.depot_return_buffer_minutes = depot_return_buffer_minutes
        self.swap_threshold = swap_threshold
        self.logger = None
        # per-vehicle tabu lists  {vehicle_id: [loc_id, ...]}
        self._tabu: dict = {}
        super().__init__()
 
    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
 
    def get_best_action(self, state, vehicle):
        tabu = self._get_tabu(vehicle)
 
        batteries_to_swap = []
        bikes_to_pickup = []
        bikes_to_deliver = []
        onsite_repairs = []
 
        if vehicle.is_at_depot():
            bikes_to_pickup, bikes_to_deliver = self._depot_actions(vehicle)
        else:
            onsite_repairs, bikes_to_pickup, bikes_to_deliver, batteries_to_swap = \
                self._station_actions(state, vehicle)

        maintenance_time = len(onsite_repairs) * MAINTENANCE_REPAIR
        service_time = sim.Action(
            batteries_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            vehicle.location.id,
            maintenance_time=maintenance_time,
            onsite_repairs=onsite_repairs,
        ).get_action_time(0.0)

        next_loc = self._choose_next(
            state,
            vehicle,
            tabu,
            len(bikes_to_pickup) - len(bikes_to_deliver),
            service_time,
        )
        self._update_tabu(vehicle, vehicle.location.id)
 
        action = sim.Action(
            batteries_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_loc,
            maintenance_time=maintenance_time,
            onsite_repairs=onsite_repairs,
        )
        self._log_to_run_logger(state, vehicle, action)
        return action
 
    # ------------------------------------------------------------------
    # Action building
    # ------------------------------------------------------------------
 
    def _depot_actions(self, vehicle):
        """At depot: drop off broken bikes, pick up repaired bikes."""
        depot = vehicle.location
 
        # Drop off every depot-flagged bike on the vehicle
        deliver = [
            b.bike_id for b in vehicle.get_bike_inventory()
            if getattr(b, "damage_status", None) == "depot"
        ]
 
        # Pick up all repaired bikes from fixed_queue (up to remaining capacity)
        remaining_cap = vehicle.bike_inventory_capacity - (
            len(vehicle.get_bike_inventory()) - len(deliver)
        )
        repaired = list(getattr(depot, "fixed_queue", {}).keys())[:remaining_cap]
 
        return repaired, deliver
 
    def _station_actions(self, state, vehicle):
        """At a regular station: repair → pickup broken → rebalance → swap."""
        loc = vehicle.location
        cap = vehicle.bike_inventory_capacity
        inv = vehicle.get_bike_inventory()
        remaining_cap = cap - len(inv)
 
        # Categorise station bikes
        onsite_bikes = [
            b for b in loc.bikes.values()
            if getattr(b, "damage_status", None) == "onsite"
            and not getattr(b, "onsite_repair_in_progress", False)
        ]
        depot_bikes = [
            b for b in loc.bikes.values()
            if getattr(b, "damage_status", None) == "depot"
        ]
        functional_station = [
            b for b in loc.bikes.values()
            if getattr(b, "damage_status", None) not in ("onsite", "depot")
        ]
        functional_vehicle = [
            b for b in inv
            if getattr(b, "damage_status", None) not in ("onsite", "depot")
        ]
 
        # 1. On-site repairs (all, no capacity consumed)
        onsite_repairs = [b.bike_id for b in onsite_bikes]
 
        # 2. Pick up depot-flagged bikes (subject to capacity)
        pickup_broken = [b.bike_id for b in depot_bikes[:remaining_cap]]
        remaining_cap -= len(pickup_broken)
 
        # 3. Rebalancing
        target = round(loc.get_target_state(state.day(), state.hour()))
        num_functional_at_station = len(functional_station)
        num_functional_on_vehicle = len(functional_vehicle)
 
        pickup_functional = []
        deliver_functional = []
 
        if num_functional_at_station < target:
            # Deliver bikes to reach target
            can_deliver = min(num_functional_on_vehicle, target - num_functional_at_station)
            deliver_functional = [b.bike_id for b in functional_vehicle[:can_deliver]]
        elif num_functional_at_station > target:
            # Pick up excess bikes
            excess = num_functional_at_station - target
            can_pickup = min(excess, remaining_cap)
            pickup_functional = [b.bike_id for b in functional_station[:can_pickup]]
 
        # 4. Battery swaps when delivering
        batteries_to_swap = []
        if deliver_functional:
            swappable = loc.get_swappable_bikes()
            n_swaps = min(vehicle.battery_inventory, len(swappable))
            batteries_to_swap = [b.bike_id for b in swappable[:n_swaps]]
 
        return (
            onsite_repairs,
            pickup_broken + pickup_functional,
            deliver_functional,
            batteries_to_swap,
        )
 
    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
 
    def _choose_next(self, state, vehicle, tabu, net_pickup_delta: int, service_time: float = 0.0):
        """
        Score every non-tabu location and return the best one.
        Routes to depot if vehicle carries many depot-bound bikes.
        """
        depot_bikes_on_vehicle = sum(
            1 for b in vehicle.get_bike_inventory()
            if getattr(b, "damage_status", None) == "depot"
        )
        force_depot = (
            depot_bikes_on_vehicle
            >= self.depot_load_threshold * vehicle.bike_inventory_capacity
        )
 
        depots = list(state.depots.values())
        nearest_depot = None
        if depots:
            nearest_depot = min(
                depots,
                key=lambda d: state.get_vehicle_travel_time(vehicle.location.id, d.id),
            )
 
        if force_depot and depots and not vehicle.is_at_depot():
            return nearest_depot.id
 
        # Build candidate set: all locations except tabu
        candidates = [
            loc_id for loc_id in state.locations
            if loc_id not in tabu and loc_id != vehicle.location.id
        ]
 
        # Fall back to all non-current locations if tabu blocks everything
        if not candidates:
            candidates = [
                loc_id for loc_id in state.locations
                if loc_id != vehicle.location.id
            ]
 
        if not candidates:
            return vehicle.location.id

        if nearest_depot is not None:
            feasible_candidates = [
                loc_id for loc_id in candidates
                if self._can_still_return_to_depot(state, vehicle, loc_id, nearest_depot.id, service_time)
            ]
            if feasible_candidates:
                candidates = feasible_candidates
            elif not vehicle.is_at_depot():
                return nearest_depot.id
            else:
                return vehicle.location.id
 
        # Compute travel times (needed for normalisation)
        travel_times = {
            loc_id: state.get_vehicle_travel_time(vehicle.location.id, loc_id)
            for loc_id in candidates
        }
        max_tt = max(travel_times.values()) or 1.0
 
        scores = {}
        for loc_id in candidates:
            loc = state.locations[loc_id]
            tt = travel_times[loc_id]
 
            # Maintenance urgency: onsite bikes + depot bikes waiting
            onsite_count = sum(
                1 for b in loc.bikes.values()
                if getattr(b, "damage_status", None) == "onsite"
                and not getattr(b, "onsite_repair_in_progress", False)
            )
            depot_count = sum(
                1 for b in loc.bikes.values()
                if getattr(b, "damage_status", None) == "depot"
            )
            maintenance_score = onsite_count + depot_count
 
            # Rebalancing need
            if hasattr(loc, "get_target_state"):
                target = loc.get_target_state(state.day(), state.hour())
                num_bikes = len([
                    b for b in loc.bikes.values()
                    if getattr(b, "damage_status", None) not in ("onsite", "depot")
                ])
                dev = abs(num_bikes - target)
 
                # Only score stations we can actually help
                bikes_on_vehicle = len([
                    b for b in vehicle.get_bike_inventory()
                    if getattr(b, "damage_status", None) not in ("onsite", "depot")
                ])
                can_help = (num_bikes < target and bikes_on_vehicle > 0) or \
                           (num_bikes > target)
                rebalancing_score = dev if can_help else 0.0
            else:
                rebalancing_score = 0.0
 
            travel_penalty = tt / max_tt  # [0, 1], higher = farther away
 
            scores[loc_id] = (
                self.w_maintenance * maintenance_score
                + self.w_rebalancing * rebalancing_score
                - self.w_travel * travel_penalty
            )
 
        return max(scores, key=scores.__getitem__)

    def _can_still_return_to_depot(self, state, vehicle, loc_id: str, depot_id: str, service_time: float) -> bool:
        """Allow loc_id only if the vehicle can still reach depot before close plus buffer."""
        if loc_id == depot_id:
            return True

        close_min = SERVICE_TIME_TO * 60.0
        clock_min = state.time % 1440.0
        time_remaining = max(0.0, close_min - clock_min)

        travel_to_next = state.get_vehicle_travel_time(vehicle.location.id, loc_id)
        travel_next_to_depot = state.get_vehicle_travel_time(loc_id, depot_id)
        required_time = (
            travel_to_next
            + service_time
            + travel_next_to_depot
            + self.depot_return_buffer_minutes
        )
        return time_remaining >= required_time

    def _log_to_run_logger(self, state, vehicle, action) -> None:
        """Write one greedy-maintenance vehicle decision to the centralized run logger."""
        logger = self.logger
        if logger is None:
            return

        current_time = state.time
        clock_min = current_time % (24 * 60)
        day = int(current_time // (24 * 60))
        clock_hour = int(clock_min // 60)
        minute = int(clock_min % 60)

        inv = vehicle.get_bike_inventory()
        func_before = sum(1 for b in inv if getattr(b, "damage_status", None) not in ("depot", "onsite"))
        depot_before = sum(1 for b in inv if getattr(b, "damage_status", None) == "depot")
        total_before = len(inv)

        raw_bikes = getattr(vehicle.location, "bikes", {})
        station_bikes = (
            raw_bikes if isinstance(raw_bikes, dict)
            else {getattr(b, "bike_id", getattr(b, "id")): b for b in raw_bikes}
        )

        is_at_depot = vehicle.is_at_depot()
        pick_up_ids = list(getattr(action, "pick_ups", []) or [])
        delivery_ids = list(getattr(action, "delivery_bikes", []) or [])

        func_pickups = 0
        depot_pickups = 0
        load_from_queue = 0
        if is_at_depot:
            fixed_queue = getattr(vehicle.location, "fixed_queue", {})
            load_from_queue = sum(1 for b_id in pick_up_ids if b_id in fixed_queue)
        else:
            for b_id in pick_up_ids:
                bike = station_bikes.get(b_id)
                if bike and getattr(bike, "damage_status", None) == "depot":
                    depot_pickups += 1
                elif bike:
                    func_pickups += 1

        vehicle_bikes = {getattr(b, "bike_id", getattr(b, "id")): b for b in inv}
        if is_at_depot:
            depot_deliveries = sum(
                1 for b_id in delivery_ids
                if getattr(vehicle_bikes.get(b_id), "damage_status", None) == "depot"
            )
            func_deliveries = 0
        else:
            depot_deliveries = 0
            func_deliveries = len(delivery_ids)

        onsite_repairs = len(getattr(action, "onsite_repairs", []) or [])
        if is_at_depot:
            func_after = func_before + load_from_queue
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

        n_damaged = sum(1 for b in station_bikes.values() if getattr(b, "damage_status", None) is not None)
        maint_flag = (n_damaged > 0) or is_at_depot
        is_maint = (onsite_repairs > 0) or (depot_pickups > 0) or (depot_deliveries > 0) or (load_from_queue > 0)

        logger.log_decision({
            "day": day,
            "hour": clock_hour,
            "minute": minute,
            "vehicle_id": vehicle.id,
            "vehicle_policy_type": self.__class__.__name__,
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
            "bikes_involved": str(pick_up_ids + delivery_ids + list(getattr(action, "onsite_repairs", []) or [])),
            "action_duration_min": round(action_duration, 3),
            "next_station_id": dest,
            "travel_time_min": round(travel_time, 3),
            "functional_load_after": func_after,
            "depot_load_after": depot_after,
            "total_load_after": total_after,
            "immediate_reward": "",
            "vfa_value": "",
            "accumulated_rollout_reward": "",
            "tail_value": "",
            "final_decision_score": "",
            "maintenance_flag_present": maint_flag,
            "selected_action_is_maintenance": is_maint,
            "n_total_candidates": "",
            "vfa_top1_next_station": "",
            "rollout_changed_decision": "",
            "winning_candidate_rank": "",
            "decision_runtime_s": "",
            "winning_profile_type": "greedy-maintenance",
        })
 
    # ------------------------------------------------------------------
    # Tabu helpers
    # ------------------------------------------------------------------
 
    def _get_tabu(self, vehicle) -> list:
        return self._tabu.get(vehicle.id, [])
 
    def _update_tabu(self, vehicle, loc_id: str):
        history = self._tabu.get(vehicle.id, [])
        history.append(loc_id)
        self._tabu[vehicle.id] = history[-self.tabu_size:]
 
 
"""
Greedy maintenance-aware baseline policy.
 
Priority order at each visit:
  1. On-site repairs (all fixable bikes at current station)
  2. Depot bike pick-up / drop-off (broken bikes in, repaired bikes out)
  3. Rebalancing toward target state
  4. Battery swaps (when delivering)
 
Routing: score every candidate station on maintenance urgency + rebalancing need,
penalised by travel time. Tabu keeps the vehicle from immediately backtracking.
"""
 
from policies import Policy
import sim
from settings import *
 
 
class _GreedyMaintenancePolicyWithoutDepotReturn(Policy):
    """
    Strong greedy baseline that aggressively handles maintenance
    while still rebalancing toward target state.
    """
 
    def __init__(
        self,
        tabu_size: int = 1,
        w_maintenance: float = 1.0,
        w_rebalancing: float = 1.0,
        w_travel: float = 0.1,
        depot_load_threshold: float = 0.5,
        swap_threshold=BATTERY_LIMIT_TO_SWAP,
    ):
        """
        w_maintenance   : weight for stations with broken/onsite bikes
        w_rebalancing   : weight for deviation from target state
        w_travel        : weight penalising far-away stations
        depot_load_threshold : fraction of vehicle capacity that, when filled
                               with depot bikes, triggers a depot trip
        """
        self.tabu_size = tabu_size
        self.w_maintenance = w_maintenance
        self.w_rebalancing = w_rebalancing
        self.w_travel = w_travel
        self.depot_load_threshold = depot_load_threshold
        self.swap_threshold = swap_threshold
        # per-vehicle tabu lists  {vehicle_id: [loc_id, ...]}
        self._tabu: dict = {}
        super().__init__()
 
    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
 
    def get_best_action(self, state, vehicle):
        tabu = self._get_tabu(vehicle)
 
        batteries_to_swap = []
        bikes_to_pickup = []
        bikes_to_deliver = []
        onsite_repairs = []
 
        if vehicle.is_at_depot():
            bikes_to_pickup, bikes_to_deliver = self._depot_actions(vehicle)
        else:
            onsite_repairs, bikes_to_pickup, bikes_to_deliver, batteries_to_swap = \
                self._station_actions(state, vehicle)
 
        next_loc = self._choose_next(state, vehicle, tabu,
                                     len(bikes_to_pickup) - len(bikes_to_deliver))
        self._update_tabu(vehicle, vehicle.location.id)
 
        return sim.Action(
            batteries_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_loc,
            maintenance_time=len(onsite_repairs) * MAINTENANCE_REPAIR,
            onsite_repairs=onsite_repairs,
        )
 
    # ------------------------------------------------------------------
    # Action building
    # ------------------------------------------------------------------
 
    def _depot_actions(self, vehicle):
        """At depot: drop off broken bikes, pick up repaired bikes."""
        depot = vehicle.location
 
        # Drop off every depot-flagged bike on the vehicle
        deliver = [
            b.bike_id for b in vehicle.get_bike_inventory()
            if getattr(b, "damage_status", None) == "depot"
        ]
 
        # Pick up all repaired bikes from fixed_queue (up to remaining capacity)
        remaining_cap = vehicle.bike_inventory_capacity - (
            len(vehicle.get_bike_inventory()) - len(deliver)
        )
        repaired = list(getattr(depot, "fixed_queue", {}).keys())[:remaining_cap]
 
        return repaired, deliver
 
    def _station_actions(self, state, vehicle):
        """At a regular station: repair → pickup broken → rebalance → swap."""
        loc = vehicle.location
        cap = vehicle.bike_inventory_capacity
        inv = vehicle.get_bike_inventory()
        remaining_cap = cap - len(inv)
 
        # Categorise station bikes
        onsite_bikes = [
            b for b in loc.bikes.values()
            if getattr(b, "damage_status", None) == "onsite"
            and not getattr(b, "onsite_repair_in_progress", False)
        ]
        depot_bikes = [
            b for b in loc.bikes.values()
            if getattr(b, "damage_status", None) == "depot"
        ]
        functional_station = [
            b for b in loc.bikes.values()
            if getattr(b, "damage_status", None) not in ("onsite", "depot")
        ]
        functional_vehicle = [
            b for b in inv
            if getattr(b, "damage_status", None) not in ("onsite", "depot")
        ]
 
        # 1. On-site repairs (all, no capacity consumed)
        onsite_repairs = [b.bike_id for b in onsite_bikes]
 
        # 2. Pick up depot-flagged bikes (subject to capacity)
        pickup_broken = [b.bike_id for b in depot_bikes[:remaining_cap]]
        remaining_cap -= len(pickup_broken)
 
        # 3. Rebalancing
        target = round(loc.get_target_state(state.day(), state.hour()))
        num_functional_at_station = len(functional_station)
        num_functional_on_vehicle = len(functional_vehicle)
 
        pickup_functional = []
        deliver_functional = []
 
        if num_functional_at_station < target:
            # Deliver bikes to reach target
            can_deliver = min(num_functional_on_vehicle, target - num_functional_at_station)
            deliver_functional = [b.bike_id for b in functional_vehicle[:can_deliver]]
        elif num_functional_at_station > target:
            # Pick up excess bikes
            excess = num_functional_at_station - target
            can_pickup = min(excess, remaining_cap)
            pickup_functional = [b.bike_id for b in functional_station[:can_pickup]]
 
        # 4. Battery swaps when delivering
        batteries_to_swap = []
        if deliver_functional:
            swappable = loc.get_swappable_bikes()
            n_swaps = min(vehicle.battery_inventory, len(swappable))
            batteries_to_swap = [b.bike_id for b in swappable[:n_swaps]]
 
        return (
            onsite_repairs,
            pickup_broken + pickup_functional,
            deliver_functional,
            batteries_to_swap,
        )
 
    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
 
    def _choose_next(self, state, vehicle, tabu, net_pickup_delta: int):
        """
        Score every non-tabu location and return the best one.
        Routes to depot if vehicle carries many depot-bound bikes.
        """
        depot_bikes_on_vehicle = sum(
            1 for b in vehicle.get_bike_inventory()
            if getattr(b, "damage_status", None) == "depot"
        )
        force_depot = (
            depot_bikes_on_vehicle
            >= self.depot_load_threshold * vehicle.bike_inventory_capacity
        )
 
        depots = list(state.depots.values())
 
        if force_depot and depots:
            best_depot = min(
                depots,
                key=lambda d: state.get_vehicle_travel_time(vehicle.location.id, d.id),
            )
            return best_depot.id
 
        # Build candidate set: all locations except tabu
        candidates = [
            loc_id for loc_id in state.locations
            if loc_id not in tabu and loc_id != vehicle.location.id
        ]
 
        # Fall back to all non-current locations if tabu blocks everything
        if not candidates:
            candidates = [
                loc_id for loc_id in state.locations
                if loc_id != vehicle.location.id
            ]
 
        if not candidates:
            return vehicle.location.id
 
        # Compute travel times (needed for normalisation)
        travel_times = {
            loc_id: state.get_vehicle_travel_time(vehicle.location.id, loc_id)
            for loc_id in candidates
        }
        max_tt = max(travel_times.values()) or 1.0
 
        scores = {}
        for loc_id in candidates:
            loc = state.locations[loc_id]
            tt = travel_times[loc_id]
 
            # Maintenance urgency: onsite bikes + depot bikes waiting
            onsite_count = sum(
                1 for b in loc.bikes.values()
                if getattr(b, "damage_status", None) == "onsite"
                and not getattr(b, "onsite_repair_in_progress", False)
            )
            depot_count = sum(
                1 for b in loc.bikes.values()
                if getattr(b, "damage_status", None) == "depot"
            )
            maintenance_score = onsite_count + depot_count
 
            # Rebalancing need
            if hasattr(loc, "get_target_state"):
                target = loc.get_target_state(state.day(), state.hour())
                num_bikes = len([
                    b for b in loc.bikes.values()
                    if getattr(b, "damage_status", None) not in ("onsite", "depot")
                ])
                dev = abs(num_bikes - target)
 
                # Only score stations we can actually help
                bikes_on_vehicle = len([
                    b for b in vehicle.get_bike_inventory()
                    if getattr(b, "damage_status", None) not in ("onsite", "depot")
                ])
                can_help = (num_bikes < target and bikes_on_vehicle > 0) or \
                           (num_bikes > target)
                rebalancing_score = dev if can_help else 0.0
            else:
                rebalancing_score = 0.0
 
            travel_penalty = tt / max_tt  # [0, 1], higher = farther away
 
            scores[loc_id] = (
                self.w_maintenance * maintenance_score
                + self.w_rebalancing * rebalancing_score
                - self.w_travel * travel_penalty
            )
 
        return max(scores, key=scores.__getitem__)
 
    # ------------------------------------------------------------------
    # Tabu helpers
    # ------------------------------------------------------------------
 
    def _get_tabu(self, vehicle) -> list:
        return self._tabu.get(vehicle.id, [])
 
    def _update_tabu(self, vehicle, loc_id: str):
        history = self._tabu.get(vehicle.id, [])
        history.append(loc_id)
        self._tabu[vehicle.id] = history[-self.tabu_size:]
 
 
