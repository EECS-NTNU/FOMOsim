import numpy as np
from Input.preprocess import get_index


class MasterParameters:

    def __init__(self, route_pattern, subproblem_scores, customer_scenarios, station_objects):

        self.route_pattern = route_pattern
        self.station_objects = station_objects

        # sets
        self.stations = [i for i in range(len(station_objects))]
        self.swap_stations = list()
        for i in range(len(station_objects)):
            if not station_objects[i].depot:
                self.swap_stations.append(i)
        self.vehicles = [i for i in range(len(self.route_pattern))]
        self.routes = [[i for i in range(len(gen.finished_gen_routes))] for gen in self.route_pattern]
        self.patterns = [[i for i in range(len(gen.patterns))] for gen in self.route_pattern]
        self.scenarios = [i for i in range(len(customer_scenarios))]
        self.starting_stations = list()

        self.origin_matrix = list()
        self.create_A_matrix()
        self.subproblem_objectives = subproblem_scores
        self.scenario_probabilities = [1/len(customer_scenarios) for i in range(len(customer_scenarios))]

        # single_pattern = Q_B, Q_CCL, Q_CCU, Q_FCL, Q_FCU
        self.pattern_b = list()
        self.pattern_ccl = list()
        self.pattern_ccu = list()
        self.pattern_fcl = list()
        self.pattern_fcu = list()
        self.set_pattern()

        self.vehicle_bike_caps = [gen.vehicle.bike_capacity for gen in self.route_pattern]
        self.station_caps = [station.station_cap for station in station_objects]
        self.init_battery_load = [gen.vehicle.current_batteries for gen in self.route_pattern]
        self.init_charged_bike_load = [gen.vehicle.current_charged_bikes for gen in self.route_pattern]
        self.init_flat_bike_load = [gen.vehicle.current_flat_bikes for gen in self.route_pattern]

        self.init_charged_station_load = [station.current_charged_bikes for station in station_objects]
        self.init_flat_station_load = [station.current_flat_bikes for station in station_objects]

        # self.print_master_params()

    """
    Create pattern variables with shape (v, r, p)
    """
    def set_pattern(self):
        for gen in self.route_pattern:
            v_pattern_b = list()
            v_pattern_ccl = list()
            v_pattern_ccu = list()
            v_pattern_fcl = list()
            v_pattern_fcu = list()
            for r in range(len(gen.finished_gen_routes)):
                v_pattern_b.append([i[0] for i in gen.patterns])
                v_pattern_ccl.append([i[1] for i in gen.patterns])
                v_pattern_fcl.append([i[2] for i in gen.patterns])
                v_pattern_ccu.append([i[3] for i in gen.patterns])
                v_pattern_fcu.append([i[4] for i in gen.patterns])
            self.pattern_b.append(v_pattern_b)
            self.pattern_ccl.append(v_pattern_ccl)
            self.pattern_ccu.append(v_pattern_ccu)
            self.pattern_fcl.append(v_pattern_fcl)
            self.pattern_fcu.append(v_pattern_fcu)

    """
    Create A-matrix with shape (v, r, i)
    """
    def create_A_matrix(self):
        for gen in self.route_pattern:
            self.starting_stations.append(get_index(gen.finished_gen_routes[0].stations[0].id, self.station_objects))
            v_row = list()
            for route in gen.finished_gen_routes:
                next_station = route.stations[1]
                index = get_index(next_station.id, self.station_objects)
                i_array = np.zeros(len(self.station_objects))
                i_array[index] = 1
                v_row.append(i_array)
            self.origin_matrix.append(v_row)

    def print_master_params(self):
        print("Route pattern: ", self.route_pattern)
        print("Station objects: ", self.station_objects)
        print("Stations: ", self.stations)
        print("Vehicles: ", self.vehicles)
        print("Starting stations: ", self.starting_stations)
        print("Routes: ", self.routes)
        print("Patterns: ", self.patterns)
        print("Scenarios: ", self.scenarios)
        print("Origin matrix: ", self.origin_matrix)
        print("Objectives: ", self.subproblem_objectives)
        print("Scenario prob: ", self.scenario_probabilities)
        print("Scenarios: ", self.scenarios)

        print("---- Patterns ----")
        print("B: ", self.pattern_b)
        print("CCL: ", self.pattern_ccl)
        print("CCU: ", self.pattern_ccu)
        print("FCL: ", self.pattern_fcl)
        print("FCU: ", self.pattern_fcu)

        print("Init battery load: ", self.init_battery_load)
        print("Vehicle bike cap: ", self.vehicle_bike_caps)
        print("Init charged bike load: ", self.init_charged_bike_load)
        print("Init flat bike load: ", self.init_flat_bike_load)

        print("Stations caps: ", self.station_caps)
        print("Init charged station bike load: ", self.init_charged_station_load)
        print("Init flat station bike load: ", self.init_flat_station_load)
