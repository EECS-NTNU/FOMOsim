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
            driving_time = get_station_car_travel_time(simul.state, vehicle.current_location, st.id)
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
            vehicle_current_location_current_flat_bikes = len(vehicle.current_location.get_swappable_scooters(settings.BATTERY_LIMIT))
            vehicle_current_station_current_charged_bikes = len(vehicle.current_location.scooters) - vehicle_current_location_current_flat_bikes
            vehicle_available_bike_capacity = vehicle.scooter_inventory_capacity - len(vehicle.scooter_inventory)
            vehicle_current_charged_bikes = len(vehicle.scooter_inventory)
            vehicle_current_location_available_parking = vehicle.current_location.capacity - len(vehicle.current_location.scooters);

            swaps = min(vehicle_current_batteries, vehicle_current_location_current_flat_bikes)

            scooters_to_swap = [scooter.id for scooter in vehicle.current_location.get_swappable_scooters()[0:swaps]]

            bat_load, bat_unload, flat_load, flat_unload = (0, 0, 0, 0)
            if vehicle_current_station_current_charged_bikes - vehicle.current_location.get_ideal_state(simul.day(), simul.hour()) > 0:
                bat_load = max(0, min(vehicle_current_station_current_charged_bikes,
                                      vehicle_available_bike_capacity,
                                      vehicle_current_station_current_charged_bikes - vehicle.current_location.get_ideal_state(simul.day(), simul.hour())))
                scooters_to_pickup = [scooter.id for scooter in vehicle.current_location.get_available_scooters()[0:bat_load]]

            else:
                bat_unload = max(0,
                                 min(vehicle_current_charged_bikes, vehicle_current_location_available_parking,
                                     vehicle.current_location.get_ideal_state(simul.day(), simul.hour()) - vehicle_current_station_current_charged_bikes))
                scooters_to_deliver = [scooter.id for scooter in vehicle.scooter_inventory[0:bat_unload]]

            # if vehicle.current_location.charging_station:
            #     flat_unload = min(vehicle.current_flat_bikes, vehicle_current_location.available_parking)
            #     vehicle_current_station_current_charged_bikes += flat_unload
            #     vehicle.current_flat_bikes -= flat_unload
            # else:
            #     flat_load = min(vehicle_current_location_current_flat_bikes, vehicle_available_bike_capacity)
            #     vehicle.current_location.current_flat_bikes -= flat_load
            #     vehicle.current_flat_bikes += flat_load
            #     handling_time = (swaps + flat_load + flat_unload + bat_unload + bat_load) * Event.handling_time
            #     driving_time = cand_scores[0][1]
            #     vehicle.current_location = next_station
            #     end_time = self.env.current_time + driving_time + handling_time + Event.parking_time
            #     self.env.trigger_stack.append(VehicleEvent(self.env.current_time, end_time, vehicle, self.env, self.greedy))
            #     self.env.trigger_stack = sorted(self.env.trigger_stack, key=lambda l: l.end_time)

        return sim.Action(
            scooters_to_swap,
            scooters_to_pickup,
            scooters_to_deliver,
            next_station.id,
        )

    def heuristic_solve(self, simul, vehicle):
        heuristic_man = HeuristicManager(simul, simul.state.vehicles, simul.state.locations,
                                         no_scenarios=10, init_branching=3,
                                         weights=(0.6, 0.1, 0.3, 0.8, 0.2), crit_weights=(0.2, 0.1, 0.5, 0.2))

        # Index of vehicle that triggered event
        next_station, pattern = heuristic_man.return_solution(vehicle_index=vehicle.id)

        driving_time = get_station_car_travel_time(simul.state, vehicle.current_location, next_station.id)
        net_charged = abs(pattern[1] - pattern[3])
        net_flat = abs(pattern[2] - pattern[4])
        handling_time = (pattern[0] + net_charged + net_flat) * EVENT_HANDLING_TIME
        end_time = driving_time + handling_time + simul.time + EVENT_PARKING_TIME

        # self.env.vehicle_vis[vehicle.id][0].append(next_station.id)
        # self.env.vehicle_vis[vehicle.id][1].append([vehicle.current_charged_bikes,
        #                                             vehicle.current_flat_bikes, vehicle.current_batteries])
        # self.env.vehicle_vis[vehicle.id][2].append(pattern)
        # self.env.vehicle_vis[vehicle.id][3].append([vehicle.current_location.current_charged_bikes,
        #                                                  vehicle.current_location.current_flat_bikes])
        return self.update_decision(simul, vehicle, vehicle.current_location, pattern, next_station)
        # self.env.trigger_stack.append(
        #     VehicleEvent(self.env.current_time, end_time, vehicle, self.env, self.greedy))
        # self.env.trigger_stack = sorted(self.env.trigger_stack, key=lambda l: l.end_time)

    def update_decision(self, simul, vehicle, station, pattern, next_station):
        Q_B, Q_CCL, Q_FCL, Q_CCU, Q_FCU = pattern[0], pattern[1], pattern[2], pattern[3], pattern[4]

        scooters_to_pickup = [ scooter.id for scooter in station.scooters[0:int(Q_CCL+Q_FCL)] ]
        scooters_to_swap = [ scooter.id for scooter in station.get_swappable_scooters() if scooter.id not in scooters_to_pickup ]
        scooters_to_deliver = [ scooter.id for scooter in vehicle.scooter_inventory[0:int(Q_CCU+Q_FCU)] ]

        # vehicle.change_battery_bikes(-Q_CCU + Q_CCL)
        # vehicle.change_flat_bikes(-Q_FCU + Q_FCL)
        # vehicle.swap_batteries(Q_B)
        # vehicle.current_location = next_station
        # station.change_charged_load(-Q_CCL + Q_CCU + Q_B)
        # if station.charging_station:  # If charging station, make flat unload battery unload immediately
        #     station.change_charged_load(-Q_FCL + Q_FCU)
        # else:
        #     station.change_flat_load(-Q_FCL + Q_FCU - Q_B)
        # if station.depot:
        #     vehicle.current_batteries = vehicle.battery_capacity

        scooters_to_swap = scooters_to_swap[0:int(Q_B)]
        scooters_to_pickup = scooters_to_pickup[0:(vehicle.battery_inventory-len(scooters_to_swap))]

        return sim.Action(
            scooters_to_swap,
            scooters_to_pickup,
            scooters_to_deliver,
            next_station.id,
        )
