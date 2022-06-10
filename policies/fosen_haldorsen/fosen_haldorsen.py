"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import settings
from policies.fosen_haldorsen.heuristic_manager import *

EVENT_HANDLING_TIME = 0.5
EVENT_PARKING_TIME = 1

class FosenHaldorsenPolicy(Policy):
    def __init__(self, greedy=False):
        self.greedy = greedy
        super().__init__()

    def get_best_action(self, simul, vehicle):
        if self.greedy:
            return self.greedy_solve(simul, vehicle)
        else:
            return self.heuristic_solve(simul, vehicle)

    def greedy_solve(self, simul, vehicle):
        no_vehicles = len(simul.state.vehicles)
        splits = len(simul.state.stations) // no_vehicles
        next_st_candidates = simul.state.stations[vehicle.id*splits:(vehicle.id+1)*splits] + simul.state.depots

        # Choose next station
        # Calculate criticality score for all stations
        cand_scores = list()
        for st in next_st_candidates:
            if st.id == vehicle.current_location.id:
                continue
            first = True
            driving_time = simul.state.get_van_travel_time(vehicle.current_location.id, st.id)
            score = get_criticality_score(simul, st, vehicle, 25, driving_time, 0.2, 0.1, 0.5, 0.2, first)
            cand_scores.append([st, driving_time, score])

        # Sort candidates by criticality score
        cand_scores = sorted(cand_scores, key=lambda l: l[2], reverse=True)

        next_station = cand_scores[0][0]

        scooters_to_swap = []
        scooters_to_pickup = []
        scooters_to_deliver = []

        if not vehicle.is_at_depot():
            # convert from new sim
            vehicle_current_batteries = vehicle.battery_inventory
            vehicle_current_station_current_flat_bikes = len(vehicle.current_location.get_swappable_scooters(settings.BATTERY_LIMIT))
            vehicle_current_station_current_charged_bikes = len(vehicle.current_location.scooters) - vehicle_current_station_current_flat_bikes
            vehicle_available_bike_capacity = vehicle.scooter_inventory_capacity - len(vehicle.scooter_inventory)
            vehicle_current_charged_bikes = len(vehicle.scooter_inventory)
            vehicle_current_location_available_parking = vehicle.current_location.capacity - len(vehicle.current_location.scooters);

            if vehicle_current_station_current_charged_bikes - vehicle.current_location.get_target_state(simul.day(), simul.hour()) > 0:
                bat_load = max(0, min(vehicle_available_bike_capacity,
                                      vehicle_current_station_current_charged_bikes - vehicle.current_location.get_target_state(simul.day(), simul.hour())))
                scooters_by_battery = sorted(vehicle.current_location.scooters, key=lambda scooter: scooter.battery, reverse=True)
                scooters_to_pickup = [scooter.id for scooter in scooters_by_battery[0:int(bat_load)]]

            else:
                bat_unload = max(0,
                                 min(vehicle_current_charged_bikes, vehicle_current_location_available_parking,
                                     vehicle.current_location.get_target_state(simul.day(), simul.hour()) - vehicle_current_station_current_charged_bikes))
                scooters_to_deliver = [scooter.id for scooter in vehicle.scooter_inventory[0:int(bat_unload)]]

            # picked up bikes low on battery get new battery, make sure we dont pick up more than we have batteries for
            swaps_for_pickups = 0
            for s in scooters_to_pickup:
                if vehicle.current_location.get_scooter_from_id(s).battery < 70:
                    swaps_for_pickups += 1
            scooters_to_pickup = scooters_to_pickup[0:(vehicle.battery_inventory-swaps_for_pickups)]

            swaps = min(vehicle_current_batteries - swaps_for_pickups, vehicle_current_station_current_flat_bikes)

            scooters_to_swap = [scooter.id for scooter in vehicle.current_location.get_swappable_scooters() if scooter.id not in scooters_to_pickup ][0:swaps]

        return sim.Action(
            scooters_to_swap,
            scooters_to_pickup,
            scooters_to_deliver,
            next_station.id,
        )

    def heuristic_solve(self, simul, vehicle):
        heuristic_man = HeuristicManager(simul, simul.state.vehicles, simul.state.locations,
                                         no_scenarios=2, init_branching=7,
                                         weights=(0.6, 0.1, 0.3, 0.8, 0.2), crit_weights=(0.2, 0.1, 0.5, 0.2))

        # Index of vehicle that triggered event
        next_station, pattern = heuristic_man.return_solution(vehicle_index=vehicle.id)

        Q_B, Q_CCL, Q_FCL, Q_CCU, Q_FCU = pattern[0], pattern[1], pattern[2], pattern[3], pattern[4]

        scooters_by_battery = sorted(vehicle.current_location.scooters, key=lambda scooter: scooter.battery, reverse=True)

        scooters_to_pickup = [ scooter.id for scooter in scooters_by_battery[0:int(Q_CCL+Q_FCL)] ]
        scooters_to_deliver = [ scooter.id for scooter in vehicle.scooter_inventory[0:int(Q_CCU+Q_FCU)] ]

        # picked up bikes low on battery get new battery, make sure we dont pick up more than we have batteries for
        swaps_for_pickups = 0
        for s in scooters_to_pickup:
            if vehicle.current_location.get_scooter_from_id(s).battery < 70:
                swaps_for_pickups += 1
        scooters_to_pickup = scooters_to_pickup[0:max(0, vehicle.battery_inventory-swaps_for_pickups)]

        swaps = min(max(0, vehicle.battery_inventory - swaps_for_pickups), Q_B)
        scooters_to_swap = [ scooter.id for scooter in vehicle.current_location.get_swappable_scooters() if scooter.id not in scooters_to_pickup ][0:int(swaps)]

        return sim.Action(
            scooters_to_swap,
            scooters_to_pickup,
            scooters_to_deliver,
            next_station.id,
        )
