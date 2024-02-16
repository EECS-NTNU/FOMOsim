"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import settings
from policies.fosen_haldorsen.heuristic_manager import *

class FosenHaldorsenPolicy(Policy):
    def __init__(self, scenarios=2, branching=7, time_horizon=25,
                 flexibility=3, average_handling_time=6, weights=(0.6, 0.1, 0.3, 0.8, 0.2), crit_weights=(0.2, 0.1, 0.5, 0.2), criticality=True, 
                 greedy=False):

        self.scenarios = scenarios
        self.branching = branching

        self.time_horizon = time_horizon
        self.handling_time = settings.MINUTES_PER_ACTION
        self.flexibility = flexibility
        self.average_handling_time = average_handling_time

        self.weights = weights
        self.crit_weights = crit_weights
        self.criticality = criticality

        self.greedy = greedy

        super().__init__()

    def init_sim(self, sim):
        if len(sim.state.stations) < self.branching:
            self.branching = len(sim.state.stations)

    def get_best_action(self, simul, vehicle):
        if self.greedy:
            return self.greedy_solve(simul, vehicle)
        else:
            return self.heuristic_solve(simul, vehicle)

    def greedy_solve(self, simul, vehicle):
        no_vehicles = len(simul.state.vehicles)
        splits = len(simul.state.stations) // no_vehicles
        next_st_candidates = list(simul.state.locations)[vehicle.vehicle_id*splits:(vehicle.vehicle_id+1)*splits] + list(simul.state.depots.values())

        # Choose next station
        # Calculate criticality score for all stations
        cand_scores = list()
        for st in next_st_candidates:
            if st.location_id == vehicle.location.location_id:
                continue
            first = True
            driving_time = simul.state.get_vehicle_travel_time(vehicle.location.location_id, st.location_id)
            score = get_criticality_score(simul, st, vehicle, 25, driving_time, 0.2, 0.1, 0.5, 0.2, first)
            cand_scores.append([st, driving_time, score])

        # Sort candidates by criticality score
        cand_scores = sorted(cand_scores, key=lambda l: l[2], reverse=True)

        next_station = cand_scores[0][0]

        bikes_to_swap = []
        bikes_to_pickup = []
        bikes_to_deliver = []

        if not vehicle.is_at_depot():
            # convert from new sim
            vehicle_current_batteries = vehicle.battery_inventory
            vehicle_current_station_current_flat_bikes = len(vehicle.location.get_swappable_bikes(settings.BATTERY_LIMIT))
            vehicle_current_station_current_charged_bikes = len(vehicle.location.bikes) - vehicle_current_station_current_flat_bikes
            vehicle_available_bike_capacity = vehicle.bike_inventory_capacity - len(vehicle.bike_inventory)
            vehicle_current_charged_bikes = len(vehicle.bike_inventory)
            vehicle_current_location_available_parking = vehicle.location.capacity - len(vehicle.location.bikes);

            if vehicle_current_station_current_charged_bikes - vehicle.location.get_target_state(simul.day(), simul.hour()) > 0:
                bat_load = max(0, min(vehicle_available_bike_capacity,
                                      vehicle_current_station_current_charged_bikes - vehicle.location.get_target_state(simul.day(), simul.hour())))
                bikes_by_battery = sorted(vehicle.location.get_bikes(), key=lambda bike: bike.battery, reverse=True)
                bikes_to_pickup = [bike.bike_id for bike in bikes_by_battery[0:int(bat_load)]]

            else:
                bat_unload = max(0,
                                 min(vehicle_current_charged_bikes, vehicle_current_location_available_parking,
                                     vehicle.location.get_target_state(simul.day(), simul.hour()) - vehicle_current_station_current_charged_bikes))
                bikes_to_deliver = [bike.bike_id for bike in vehicle.get_bike_inventory()[0:int(bat_unload)]]

            # picked up bikes low on battery get new battery, make sure we dont pick up more than we have batteries for
            swaps_for_pickups = 0
            for s in bikes_to_pickup:
                if vehicle.location.get_bike_from_id(s).battery < 70:
                    swaps_for_pickups += 1
            bikes_to_pickup = bikes_to_pickup[0:(vehicle.battery_inventory-swaps_for_pickups)]

            swaps = min(vehicle_current_batteries - swaps_for_pickups, vehicle_current_station_current_flat_bikes)

            bikes_to_swap = [bike.bike_id for bike in vehicle.location.get_swappable_bikes() if bike.bike_id not in bikes_to_pickup ][0:swaps]

        return sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_station.location_id,
        )

    def heuristic_solve(self, simul, vehicle):
        heuristic_man = HeuristicManager(simul, simul.state.vehicles, simul.state.locations,
                                         no_scenarios=self.scenarios, init_branching=self.branching,
                                         time_horizon=self.time_horizon, handling_time=self.handling_time, flexibility=self.flexibility,
                                         average_handling_time=self.average_handling_time, seed_scenarios_subproblems=simul.state.rng.integers(10000), 
                                         criticality=self.criticality, weights=self.weights, crit_weights=self.crit_weights)

        # Index of vehicle that triggered event
        next_station, pattern = heuristic_man.return_solution(vehicle_index=vehicle.vehicle_id)

        Q_B, Q_CCL, Q_FCL, Q_CCU, Q_FCU = pattern[0], pattern[1], pattern[2], pattern[3], pattern[4]

        bikes_by_battery = sorted(vehicle.location.get_bikes(), key=lambda bike: bike.battery, reverse=True)

        bikes_to_pickup = [ bike.bike_id for bike in bikes_by_battery[0:int(Q_CCL+Q_FCL)] ]
        bikes_to_deliver = [ bike.bike_id for bike in vehicle.get_bike_inventory()[0:int(Q_CCU+Q_FCU)] ]

        # picked up bikes low on battery get new battery, make sure we dont pick up more than we have batteries for
        swaps_for_pickups = 0
        for s in bikes_to_pickup:
            if vehicle.location.get_bike_from_id(s).battery < 70:
                swaps_for_pickups += 1
        bikes_to_pickup = bikes_to_pickup[0:max(0, vehicle.battery_inventory-swaps_for_pickups)]

        swaps = min(max(0, vehicle.battery_inventory - swaps_for_pickups), Q_B)
        bikes_to_swap = [ bike.bike_id for bike in vehicle.location.get_swappable_bikes() if bike.bike_id not in bikes_to_pickup ][0:int(swaps)]

        return sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_station.location_id,
        )
