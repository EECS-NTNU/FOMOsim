"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import numpy.random as random
import abc

class FosenHaldorsenPolicy(Policy):
    def __init__(self, greedy = False):
        super().__init__()
        self.greedy = greedy

    def get_best_action(self, world, vehicle):
        if self.greedy:
            return self.greedy_solve(world, vehicle)
        else:
            return self.heuristic_solve(world, vehicle)
            
    def greedy_solve(self, world, vehicle):
        hour = world.time // 60
        no_vehicles = len(world.state.vehicles)
        splits = len(world.state.stations) // no_vehicles
        next_st_candidates = world.state.stations[vehicle.id*splits:(vehicle.id+1)*splits]

        # Choose next station
        # Calculate criticality score for all stations
        cand_scores = list()
        for st in next_st_candidates:
            if st.id == vehicle.current_location.id:
                continue
            first = True
            driving_time = vehicle.current_location.get_station_car_travel_time(st.id)
            score = st.get_criticality_score(vehicle, 25, hour,
                                             driving_time, 0.2, 0.1, 0.5, 0.2, first)
            cand_scores.append([st, driving_time, score])

        # Sort candidates by criticality score
        cand_scores = sorted(cand_scores, key=lambda l: l[2], reverse=True)

        next_station = cand_scores[0][0]

        swaps, bat_load, bat_unload, flat_load, flat_unload = (0, 0, 0, 0, 0)
        if vehicle.current_station.depot:
            vehicle.current_batteries = vehicle.battery_capacity
        else:
            swaps = min(vehicle.current_batteries, vehicle.current_station.current_flat_bikes)
            vehicle.current_batteries -= swaps
            vehicle.current_station.current_flat_bikes -= swaps
            vehicle.current_station.current_charged_bikes += swaps
            bat_load, bat_unload, flat_load, flat_unload = (0, 0, 0, 0)
            if vehicle.current_station.current_charged_bikes - vehicle.current_station.get_ideal_state(hour) > 0:
                bat_load = max(0, min(vehicle.current_station.current_charged_bikes,
                                      vehicle.available_bike_capacity(),
                                      vehicle.current_station.current_charged_bikes - vehicle.current_station.get_ideal_state(
                                          hour)))
                vehicle.current_station.current_charged_bikes -= bat_load
                vehicle.current_charged_bikes += bat_load
            else:
                bat_unload = max(0,
                                 min(vehicle.current_charged_bikes, vehicle.current_station.available_parking(),
                                     vehicle.current_station.get_ideal_state(
                                         hour) - vehicle.current_station.current_charged_bikes))
                vehicle.current_station.current_charged_bikes += bat_unload
                vehicle.current_charged_bikes -= bat_unload
            if vehicle.current_station.charging_station:
                flat_unload = min(vehicle.current_flat_bikes, vehicle.current_station.available_parking())
                vehicle.current_station.current_charged_bikes += flat_unload
                vehicle.current_flat_bikes -= flat_unload
            else:
                flat_load = min(vehicle.current_station.current_flat_bikes, vehicle.available_bike_capacity())
                vehicle.current_station.current_flat_bikes -= flat_load
                vehicle.current_flat_bikes += flat_load
        handling_time = (swaps + flat_load + flat_unload + bat_unload + bat_load) * Event.handling_time
        driving_time = cand_scores[0][1]
        vehicle.current_station = next_station
        end_time = self.env.current_time + driving_time + handling_time + Event.parking_time
        self.env.trigger_stack.append(VehicleEvent(self.env.current_time, end_time, vehicle, self.env, self.greedy))
        self.env.trigger_stack = sorted(self.env.trigger_stack, key=lambda l: l.end_time)

        return sim.Action([], [], [], 0)

    def heuristic_solve(self, world, vehicle):
        hour = self.env.current_time // 60
        start = time.time()
        heuristic_man = HeuristicManager(self.env.vehicles, self.env.stations, hour,
                                         no_scenarios=self.env.scenarios, init_branching=self.env.init_branching,
                                         weights=self.env.weights, crit_weights=self.env.crit_weights)
        self.event_time = time.time() - start

        # Index of vehicle that triggered event
        next_station, pattern = heuristic_man.return_solution(vehicle_index=vehicle.id)

        driving_time = vehicle.current_station.get_station_car_travel_time(next_station.id)
        net_charged = abs(pattern[1] - pattern[3])
        net_flat = abs(pattern[2] - pattern[4])
        handling_time = (pattern[0] + net_charged + net_flat) * Event.handling_time
        end_time = driving_time + handling_time + self.env.current_time + Event.parking_time

        self.env.vehicle_vis[vehicle.id][0].append(next_station.id)
        self.env.vehicle_vis[vehicle.id][1].append([vehicle.current_charged_bikes,
                                                         vehicle.current_flat_bikes, vehicle.current_batteries])
        self.env.vehicle_vis[vehicle.id][2].append(pattern)
        self.env.vehicle_vis[vehicle.id][3].append([vehicle.current_station.current_charged_bikes,
                                                         vehicle.current_station.current_flat_bikes])
        self.update_decision(vehicle, vehicle.current_station, pattern, next_station)
        self.env.trigger_stack.append(
            VehicleEvent(self.env.current_time, end_time, vehicle, self.env, self.greedy))
        self.env.trigger_stack = sorted(self.env.trigger_stack, key=lambda l: l.end_time)

        return sim.Action([], [], [], 0)

    def update_decision(self, vehicle, station, pattern, next_station):
        Q_B, Q_CCL, Q_FCL, Q_CCU, Q_FCU = pattern[0], pattern[1], pattern[2], pattern[3], pattern[4]
        vehicle.change_battery_bikes(-Q_CCU + Q_CCL)
        vehicle.change_flat_bikes(-Q_FCU + Q_FCL)
        vehicle.swap_batteries(Q_B)
        vehicle.current_station = next_station
        station.change_charged_load(-Q_CCL + Q_CCU + Q_B)
        if station.charging_station:  # If charging station, make flat unload battery unload immediately
            station.change_charged_load(-Q_FCL + Q_FCU)
        else:
            station.change_flat_load(-Q_FCL + Q_FCU - Q_B)
        if station.depot:
            vehicle.current_batteries = vehicle.battery_capacity
    
