import copy
import settings
import policies.fosen_haldorsen.heuristic_manager as hm
import sim

class Route:

    def __init__(self, starting_st, vehicle, day, hour, time_hor=25):
        self.starting_station = starting_st
        self.stations = [starting_st]
        self.length = 0
        self.station_visits = [0]
        self.upper_extremes = None
        self.time_horizon = time_hor
        self.vehicle = vehicle
        self.handling_time = 0.5
        self.day = day
        self.hour = hour

    def add_station(self, station, added_station_time):
        self.stations.append(station)
        self.length += added_station_time
        self.station_visits.append(self.length)

    def generate_extreme_decisions(self, policy='greedy'):
        swap, bat_load, flat_load, bat_unload, flat_unload = (0, 0, 0, 0, 0)
        if not isinstance(self.starting_station, sim.Depot):
            if policy == 'greedy':
                # convert from new sim
                starting_station_current_charged_bikes = len(self.starting_station.get_available_scooters())
                vehicle_available_bike_capacity = self.vehicle.scooter_inventory_capacity - len(self.vehicle.scooter_inventory)
                vehicle_current_charged_bikes = len(self.vehicle.scooter_inventory)
                starting_station_available_parking = self.starting_station.capacity - len(self.starting_station.scooters)
                starting_station_current_flat_bikes = len(self.starting_station.get_swappable_scooters(settings.BATTERY_LIMIT))
                vehicle_current_flat_bikes = 0
                vehicle_current_batteries = self.vehicle.battery_inventory

                bat_load = max(0, min(starting_station_current_charged_bikes, vehicle_available_bike_capacity,
                                      starting_station_current_charged_bikes - self.starting_station.get_target_state(self.day, self.hour)))
                bat_unload = max(0, min(vehicle_current_charged_bikes, starting_station_available_parking,
                                 self.starting_station.get_target_state(self.day, self.hour) - starting_station_current_charged_bikes))
                flat_load = min(starting_station_current_flat_bikes, vehicle_available_bike_capacity)
                flat_unload = min(vehicle_current_flat_bikes, starting_station_available_parking)
                swap = min(vehicle_current_batteries,
                           starting_station_current_flat_bikes + vehicle_current_flat_bikes)
        # Q_B, Q_CCL, Q_FCL, Q_CCU, Q_FCU
        self.upper_extremes = [swap, bat_load, flat_load, bat_unload, flat_unload]


class GenerateRoutePattern:

    flexibility = 3
    average_handling_time = 6

    def __init__(self, simul, starting_st, stations, vehicle, init_branching=8, criticality=True, dynamic=True,
                 crit_weights=None):
        self.simul = simul
        self.starting_station = starting_st
        self.time_horizon = 25
        self.vehicle = vehicle
        self.finished_gen_routes = None
        self.patterns = None
        self.all_stations = stations
        self.init_branching = init_branching
        self.criticality = criticality
        self.dynamic = dynamic
        self.w_drive, self.w_dev, self.w_viol, self.w_net = crit_weights

    def get_station_car_travel_time(self, station, end_st_id):
        return self.simul.state.get_distance(station.id, end_st_id) / settings.VEHICLE_SPEED

    def get_columns(self):
        finished_routes = list()
        construction_routes = [Route(self.starting_station, self.vehicle, self.simul.day(), self.simul.hour())]
        while construction_routes:
            for col in construction_routes:
                if col.length < (self.time_horizon - GenerateRoutePattern.flexibility):
                    if not self.criticality:
                        cand_scores = col.starting_station.get_candidate_stations(
                            self.all_stations, tabu_list=[c.id for c in col.stations], max_candidates=9)
                    # candidates = all stations
                    else:
                        candidates = self.all_stations
                        cand_scores = list()

                        # Calculate criticality score for all stations
                        for st in candidates:
                            if st not in col.stations:
                                first = False
                                if len(col.stations) == 1:
                                    first = True
                                driving_time = self.get_station_car_travel_time(col.stations[-1], st.id)
                                score = hm.get_criticality_score(self.simul, st, self.vehicle, self.time_horizon, 
                                                                 driving_time, self.w_viol,
                                                                 self.w_drive, self.w_dev, self.w_net, first)
                                cand_scores.append([st, driving_time, score])

                        # Sort candidates by criticality score
                        cand_scores = sorted(cand_scores, key=lambda l: l[2], reverse=True)
                    # Filtering (remember on/off opportunity)

                    # Extend the route with the B best stations
                    for j in range(self.init_branching):
                        new_col = copy.deepcopy(col)
                        new_col.add_station(cand_scores[j][0], cand_scores[j][1] +
                                            GenerateRoutePattern.average_handling_time)
                        construction_routes.append(new_col)

                else:
                    col.generate_extreme_decisions()
                    finished_routes.append(col)
                construction_routes.remove(col)
                if self.init_branching > 4 and self.dynamic:
                    self.init_branching = 1
                elif self.init_branching > 1 and self.dynamic:
                    self.init_branching = 1
        self.finished_gen_routes = finished_routes
        self.gen_patterns()

    def gen_patterns(self):
        rep_col = self.finished_gen_routes[0]
        pat = list()
        # Q_B, Q_CCL, Q_FCL, Q_CCU, Q_FCU
        for swap in [0, rep_col.upper_extremes[0]]:
            for bat_load in [0, rep_col.upper_extremes[1]]:
                for bat_unload in [0, rep_col.upper_extremes[3]]:
                    flat_load_upper = rep_col.upper_extremes[2]
                    flat_unload_upper = rep_col.upper_extremes[4]
                    pat.append([swap, bat_load, 0, bat_unload, 0])
                    pat.append([swap // 2, bat_load // 2, 0, bat_unload // 2, 0])
                    pat.append([swap // 4, bat_load // 4, 0, bat_unload // 4, 0])
                    pat.append([swap // 4 * 3, bat_load // 4 * 3, 0, bat_unload // 4 * 3, 0])
                    pat.append([swap, bat_load, flat_load_upper, bat_unload, 0])
                    pat.append([swap // 2, bat_load // 2, flat_load_upper // 2, bat_unload // 2, 0])
                    pat.append([swap // 4, bat_load // 4, flat_load_upper // 4, bat_unload // 4, 0])
                    pat.append([swap // 4 * 3, bat_load // 4 * 3, flat_load_upper // 4 * 3, bat_unload // 4 * 3, 0])
                    pat.append([swap, bat_load, 0, bat_unload, flat_unload_upper])
                    pat.append([swap // 2, bat_load // 2, 0, bat_unload // 2, flat_unload_upper // 2])
                    pat.append([swap // 4, bat_load // 4, 0, bat_unload // 4, flat_unload_upper // 4])
                    pat.append([swap // 4 * 3, bat_load // 4 * 3, 0, bat_unload // 4 * 3, flat_unload_upper // 4 * 3])
        self.patterns = list(set(tuple(val) for val in pat))
