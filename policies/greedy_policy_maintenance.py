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
        w_maintenance: float = 3.0,
        w_rebalancing: float = 1.0,
        w_travel: float = 0.5,
        depot_load_threshold: float = 0.9,
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
