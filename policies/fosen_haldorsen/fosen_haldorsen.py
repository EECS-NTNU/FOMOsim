"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import settings

class FosenHaldorsenPolicy(Policy):
    def __init__(self, greedy=False):
        self.greedy = greedy
        super().__init__()

    def get_best_action(self, simul, vehicle):
        if self.greedy:
            return self.greedy_solve(simul, vehicle)
        else:
            return self.heuristic_solve(simul, vehicle)

    def get_criticality_score(self, simul, location, vehicle, time_horizon, driving_time, w_viol, w_drive, w_dev, w_net, first_station):
        # converting values from new simulator
        vehicle_current_batteries = vehicle.battery_inventory
        incoming_flat_bike_rate_plus_incoming_charged_bike_rate = location.get_arrive_intensity(simul.day(), simul.hour())
        location_incoming_charged_bike_rate = location.get_arrive_intensity(simul.day(), simul.hour())
        demand_per_hour = location.get_leave_intensity(simul.day(), simul.hour())
        location_current_charged_bikes = len(location.get_available_scooters())
        location_current_flat_bikes = len(location.get_swappable_scooters(settings.BATTERY_LIMIT))
        vehicle_current_station_current_charged_bikes = len(vehicle.current_location.get_available_scooters())
        location_get_incoming_charged_rate = location.get_arrive_intensity(simul.day(), simul.hour())
        location_get_outgoing_customer_rate = location.get_leave_intensity(simul.day(), simul.hour())

        # ------- Time to violation -------
        if vehicle.battery_inventory_capacity == 0 and isinstance(location, sim.Depot):
            return -100000
        if isinstance(location, sim.Depot) and vehicle_current_batteries < 2:
            return 100000
        time_to_starvation = 10000
        time_to_congestion = 10000
        # Time to congestion
        # Ensure that denominator != 0
        if (incoming_flat_bike_rate_plus_incoming_charged_bike_rate - demand_per_hour != 0):
            t_cong = (location.station_cap - location_current_charged_bikes - location_current_flat_bikes)/(
                (incoming_flat_bike_rate_plus_incoming_charged_bike_rate -
                 demand_per_hour) / 60)
            if t_cong > 0:
                time_to_congestion = t_cong
        # Time to starvation
        # Ensure that denominator != 0
        if (demand_per_hour - location_incoming_charged_bike_rate) != 0:
            t_starv = location_current_charged_bikes / ((demand_per_hour
                        - location_incoming_charged_bike_rate) / 60)
            if t_starv > 0:
                time_to_starvation = t_starv
        time_to_violation = min(time_to_starvation, time_to_congestion)
        if (vehicle_current_station_current_charged_bikes - vehicle.current_location.get_ideal_state(simul.day(), simul.hour())) > 0 and (incoming_flat_bike_rate_plus_incoming_charged_bike_rate -
                 demand_per_hour) > 0 and first_station and location_current_charged_bikes > location.get_ideal_state(simul.day(), simul.hour()):
            return -10000
        # ------- Deviation at time horizon  -------
        # Starving station
        if demand_per_hour - location_incoming_charged_bike_rate > 0:
            charged_at_t = location_current_charged_bikes - (demand_per_hour -
                    location_incoming_charged_bike_rate) * min(time_horizon, time_to_starvation)
            if location.get_ideal_state(simul.day(), simul.hour()) - charged_at_t > 0 and first_station and (vehicle.current_charged_bikes < 2 and (
                    vehicle_current_batteries < 2 or location_current_flat_bikes < 2)):
                return -10000
        # Congesting station
        elif demand_per_hour - location_incoming_charged_bike_rate < 0:
            charged_at_t = location_current_charged_bikes + (location_incoming_charged_bike_rate
                    - demand_per_hour) * min(time_horizon, time_to_congestion)
            if location.available_parking() == 0 and first_station and vehicle.available_bike_capacity() < 2:
                return -10000
        else:
            charged_at_t = location_current_charged_bikes
        dev = abs(location.get_ideal_state(simul.day(), simul.hour()) - charged_at_t)
        net = abs(location_get_incoming_charged_rate - location_get_outgoing_customer_rate)
        return - w_viol * time_to_violation - w_drive * driving_time + w_dev * dev + w_net * net

    def get_station_car_travel_time(self, state, station, end_st_id):
        return state.get_distance(station.id, end_st_id) / settings.VEHICLE_SPEED

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
            driving_time = self.get_station_car_travel_time(simul.state, vehicle.current_location, st.id)
            score = self.get_criticality_score(simul, st, vehicle, 25, driving_time, 0.2, 0.1, 0.5, 0.2, first)
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
        hour = self.env.current_time // 60
        start = time.time()
        heuristic_man = HeuristicManager(simul.state.vehicles, simul.state.stations, hour,
                                         no_scenarios=self.env.scenarios, init_branching=self.env.init_branching,
                                         weights=self.env.weights, crit_weights=self.env.crit_weights)
        self.event_time = time.time() - start

        # Index of vehicle that triggered event
        next_station, pattern = heuristic_man.return_solution(vehicle_index=vehicle.id)

        driving_time = vehicle.current_location.get_station_car_travel_time(next_station.id)
        net_charged = abs(pattern[1] - pattern[3])
        net_flat = abs(pattern[2] - pattern[4])
        handling_time = (pattern[0] + net_charged + net_flat) * Event.handling_time
        end_time = driving_time + handling_time + self.env.current_time + Event.parking_time

        self.env.vehicle_vis[vehicle.id][0].append(next_station.id)
        self.env.vehicle_vis[vehicle.id][1].append([vehicle.current_charged_bikes,
                                                         vehicle.current_flat_bikes, vehicle.current_batteries])
        self.env.vehicle_vis[vehicle.id][2].append(pattern)
        self.env.vehicle_vis[vehicle.id][3].append([vehicle.current_location.current_charged_bikes,
                                                         vehicle.current_location.current_flat_bikes])
        self.update_decision(vehicle, vehicle.current_location, pattern, next_station)
        self.env.trigger_stack.append(
            VehicleEvent(self.env.current_time, end_time, vehicle, self.env, self.greedy))
        self.env.trigger_stack = sorted(self.env.trigger_stack, key=lambda l: l.end_time)

    def update_decision(self, vehicle, station, pattern, next_station):
        Q_B, Q_CCL, Q_FCL, Q_CCU, Q_FCU = pattern[0], pattern[1], pattern[2], pattern[3], pattern[4]
        vehicle.change_battery_bikes(-Q_CCU + Q_CCL)
        vehicle.change_flat_bikes(-Q_FCU + Q_FCL)
        vehicle.swap_batteries(Q_B)
        vehicle.current_location = next_station
        station.change_charged_load(-Q_CCL + Q_CCU + Q_B)
        if station.charging_station:  # If charging station, make flat unload battery unload immediately
            station.change_charged_load(-Q_FCL + Q_FCU)
        else:
            station.change_flat_load(-Q_FCL + Q_FCU - Q_B)
        if station.depot:
            vehicle.current_batteries = vehicle.battery_capacity
