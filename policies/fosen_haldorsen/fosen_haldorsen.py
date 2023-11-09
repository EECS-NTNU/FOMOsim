"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import settings
from policies.fosen_haldorsen.heuristic_manager import *

class FosenHaldorsenPolicy(Policy):
    def __init__(
            self, 
            scenarios=2, # Number of scenarios to consider in the policy
            branching=7, # Branching factor for scenario generation
            time_horizon=25, # Time horizon over which the policy operates
            flexibility=3, # Flexibility parameter, possibly defining how adaptable the policy is
            average_handling_time=6, # Average handling time for tasks within the policy
            weights=(0.6, 0.1, 0.3, 0.8, 0.2), # Tuple of weights for different criteria in the policy
            crit_weights=(0.2, 0.1, 0.5, 0.2), # Tuple of criticality weights for prioritizing tasks
            criticality=True, # Boolean indicating whether criticality is considered
            greedy=False # Boolean indicating if a greedy approach is used
            ): 

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
        """
        Initialize the simulation with given parameter - adjust the branching if necessary.
        """
        if len(sim.state.stations) < self.branching:
            self.branching = len(sim.state.stations)

    def get_best_action(self, simul, vehicle):
        """
        Determines the action for the vehicle - uses greedy approach if self.greedy = True, otherwise heuristic approach
        """
        if self.greedy:
            return self.greedy_solve(simul, vehicle)
        else:
            return self.heuristic_solve(simul, vehicle)

    def greedy_solve(self, simul, vehicle):
        """
        Chooses the best greedy action at current state
        """
        no_vehicles = len(simul.state.vehicles) # Number of vehicles in the simulation
        splits = len(simul.state.stations) // no_vehicles # Number of stations each vehicle should handle
        next_st_candidates = list(simul.state.locations)[vehicle.id*splits:(vehicle.id+1)*splits] + list(simul.state.depots.values()) # List of potental next stations for the vehicle to consider

        # Choose next station
        # Calculate criticality score for all stations
        cand_scores = list()
        for st in next_st_candidates:
            if st.id == vehicle.location.id:
                continue
            first = True
            driving_time = simul.state.get_vehicle_travel_time(vehicle.location.id, st.id) # Calculate driving time to each candidate from current location
            score = get_criticality_score(simul, st, vehicle, 25, driving_time, 0.2, 0.1, 0.5, 0.2, first) # Calculate criticality score for the station
            cand_scores.append([st, driving_time, score])

        # Sort candidates by criticality score in descending order
        cand_scores = sorted(cand_scores, key=lambda l: l[2], reverse=True)

        # Next station is the station with highest criticality score
        next_station = cand_scores[0][0]

        bikes_to_swap = [] # Bikes to swap batteries on
        bikes_to_pickup = [] # Bikes to pick up at station
        bikes_to_deliver = [] # Bikes to unload to current station

        # Not in depot -> calculate bike and battery inventory
        if not vehicle.is_at_depot():
            # convert from new sim
            vehicle_current_batteries = vehicle.battery_inventory
            vehicle_current_station_current_flat_bikes = len(vehicle.location.get_swappable_bikes(settings.BATTERY_LIMIT))
            vehicle_current_station_current_charged_bikes = len(vehicle.location.bikes) - vehicle_current_station_current_flat_bikes
            vehicle_available_bike_capacity = vehicle.bike_inventory_capacity - len(vehicle.bike_inventory)
            vehicle_current_charged_bikes = len(vehicle.bike_inventory)
            vehicle_current_location_available_parking = vehicle.location.capacity - len(vehicle.location.bikes)

            if vehicle_current_station_current_charged_bikes - vehicle.location.get_target_state(simul.day(), simul.hour()) > 0:
                bat_load = max(0, min(vehicle_available_bike_capacity,
                                      vehicle_current_station_current_charged_bikes - vehicle.location.get_target_state(simul.day(), simul.hour())))
                bikes_by_battery = sorted(vehicle.location.get_bikes(), key=lambda bike: bike.battery, reverse=True)
                bikes_to_pickup = [bike.id for bike in bikes_by_battery[0:int(bat_load)]]

            else:
                bat_unload = max(0,
                                 min(vehicle_current_charged_bikes, vehicle_current_location_available_parking,
                                     vehicle.location.get_target_state(simul.day(), simul.hour()) - vehicle_current_station_current_charged_bikes))
                bikes_to_deliver = [bike.id for bike in vehicle.get_bike_inventory()[0:int(bat_unload)]]

            # picked up bikes low on battery get new battery, make sure we dont pick up more than we have batteries for
            swaps_for_pickups = 0
            for s in bikes_to_pickup:
                if vehicle.location.get_bike_from_id(s).battery < 70:
                    swaps_for_pickups += 1
            bikes_to_pickup = bikes_to_pickup[0:(vehicle.battery_inventory-swaps_for_pickups)]

            swaps = min(vehicle_current_batteries - swaps_for_pickups, vehicle_current_station_current_flat_bikes)

            bikes_to_swap = [bike.id for bike in vehicle.location.get_swappable_bikes() if bike.id not in bikes_to_pickup ][0:swaps]

        return sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_station.id,
        )

    def heuristic_solve(self, simul, vehicle):
        heuristic_man = HeuristicManager(simul, simul.state.vehicles, simul.state.locations,
                                         no_scenarios=self.scenarios, init_branching=self.branching,
                                         time_horizon=self.time_horizon, handling_time=self.handling_time, flexibility=self.flexibility,
                                         average_handling_time=self.average_handling_time, seed_scenarios_subproblems=simul.state.rng.integers(10000), 
                                         criticality=self.criticality, weights=self.weights, crit_weights=self.crit_weights)

        # Index of vehicle that triggered event
        next_station, pattern = heuristic_man.return_solution(vehicle_index=vehicle.id)

        Q_B, Q_CCL, Q_FCL, Q_CCU, Q_FCU = pattern[0], pattern[1], pattern[2], pattern[3], pattern[4]

        bikes_by_battery = sorted(vehicle.location.get_bikes(), key=lambda bike: bike.battery, reverse=True)

        bikes_to_pickup = [ bike.id for bike in bikes_by_battery[0:int(Q_CCL+Q_FCL)] ]
        bikes_to_deliver = [ bike.id for bike in vehicle.get_bike_inventory()[0:int(Q_CCU+Q_FCU)] ]

        # picked up bikes low on battery get new battery, make sure we dont pick up more than we have batteries for
        swaps_for_pickups = 0
        for s in bikes_to_pickup:
            if vehicle.location.get_bike_from_id(s).battery < 70:
                swaps_for_pickups += 1
        bikes_to_pickup = bikes_to_pickup[0:max(0, vehicle.battery_inventory-swaps_for_pickups)]

        swaps = min(max(0, vehicle.battery_inventory - swaps_for_pickups), Q_B)
        bikes_to_swap = [ bike.id for bike in vehicle.location.get_swappable_bikes() if bike.id not in bikes_to_pickup ][0:int(swaps)]

        return sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_station.id,
        )
