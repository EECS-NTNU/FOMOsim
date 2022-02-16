from Input.preprocess import get_index
from Subproblem.model_manager import ModelManager
from Subproblem.generate_route_pattern import GenerateRoutePattern
import numpy as np
from MasterProblem.master_params import MasterParameters
from MasterProblem.master_model import run_master_model


class HeuristicManager:

    time_h = 25

    def __init__(self, vehicles, station_full_set, hour, no_scenarios=1, init_branching=7, weights=None,
                 criticality=True, writer=None, crit_weights=None):
        self.no_scenarios = no_scenarios
        self.customer_arrival_scenarios = list()
        self.vehicles = vehicles
        self.station_set = station_full_set
        self.route_patterns = list()
        self.subproblem_scores = list()
        self.master_solution = None
        self.init_branching = init_branching
        self.hour = hour
        self.weights = weights
        self.criticality = criticality
        self.crit_weights = crit_weights
        self.writer = writer

        self.generate_scenarios()

        self.run_subproblems()
        self.run_master_problem()

    def reset_manager_and_run(self, branching):
        self.route_patterns = list()
        self.subproblem_scores = list()
        self.master_solution = None
        self.init_branching = branching
        self.run_subproblems()
        self.run_master_problem()

    def run_vehicle_subproblems(self, vehicle):
        gen = GenerateRoutePattern(vehicle.current_station, self.station_set, vehicle,
                                   self.hour, init_branching=self.init_branching, criticality=self.criticality,
                                   crit_weights=self.crit_weights)
        gen.get_columns()
        self.route_patterns.append(gen)
        model_man = ModelManager(vehicle, self.hour)
        route_scores = list()
        for route in gen.finished_gen_routes:
            route_full_set_index = [get_index(st.id, self.station_set) for st in route.stations]
            pattern_scores = list()
            for pattern in gen.patterns:
                scenario_scores = list()
                for customer_scenario in self.customer_arrival_scenarios:
                    score = model_man.run_one_subproblem(route, route_full_set_index, pattern, customer_scenario,
                                                         self.weights)
                    scenario_scores.append(score)
                pattern_scores.append(scenario_scores)
            route_scores.append(pattern_scores)
        self.subproblem_scores.append(route_scores)

    def run_subproblems(self):
        for vehicle in self.vehicles:
            self.run_vehicle_subproblems(vehicle)

    def run_master_problem(self):
        params = MasterParameters(route_pattern=self.route_patterns, subproblem_scores=self.subproblem_scores,
                                  customer_scenarios=self.customer_arrival_scenarios, station_objects=self.station_set)
        self.master_solution = run_master_model(params)

    def return_solution(self, vehicle_index):
        i = None
        q_B, q_CCL, q_FCL, q_CCU, q_FCU = 0, 0, 0, 0, 0
        for var in self.master_solution.getVars():
            name = var.varName.strip("]").split("[")
            iv = name[1].split(',')
            round_val = round(var.x, 0)
            if name[0] == 'x_nac' and round_val == 1 and int(iv[1]) == vehicle_index:
                i = int(iv[0])
            if name[0] == 'q_FCL_nac' and int(name[1]) == vehicle_index:
                q_FCL = round(var.x, 0)
            if name[0] == 'q_CCL_nac' and int(name[1]) == vehicle_index:
                q_CCL = round(var.x, 0)
            if name[0] == 'q_FCU_nac' and int(name[1]) == vehicle_index:
                q_FCU = round(var.x, 0)
            if name[0] == 'q_CCU_nac' and int(name[1]) == vehicle_index:
                q_CCU = round(var.x, 0)
            if name[0] == 'q_B_nac' and int(name[1][0]) == vehicle_index:
                q_B = round(var.x, 0)
        return self.station_set[i], [q_B, q_CCL, q_FCL, q_CCU, q_FCU]

    def generate_scenarios(self):
        for i in range(self.no_scenarios):
            scenario = list()
            for station in self.station_set:
                c1_times = HeuristicManager.poisson_simulation(station.get_incoming_charged_rate(self.hour) / 60, HeuristicManager.time_h)
                c2_times = HeuristicManager.poisson_simulation(station.get_incoming_flat_rate(self.hour) / 60, HeuristicManager.time_h)
                c3_times = HeuristicManager.poisson_simulation(station.get_outgoing_customer_rate(self.hour) / 60, HeuristicManager.time_h)
                scenario.append([c1_times, c2_times, c3_times])
            self.customer_arrival_scenarios.append(scenario)

    @staticmethod
    def poisson_simulation(intensity_rate, time_steps):
        times = list()
        for t in range(time_steps):
            arrival = np.random.poisson(intensity_rate)
            for i in range(arrival):
                times.append(t)
        return times
