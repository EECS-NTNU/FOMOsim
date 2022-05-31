from policies.fosen_haldorsen.Subproblem.model_manager import ModelManager
from policies.fosen_haldorsen.Subproblem.generate_route_pattern import GenerateRoutePattern
import numpy as np
from policies.fosen_haldorsen.MasterProblem.master_params import MasterParameters
from policies.fosen_haldorsen.MasterProblem.master_model import run_master_model
import settings
import sim

def get_index(station_id, stations):
    for i in range(len(stations)):
        if stations[i].id == station_id:
            return i

def get_criticality_score(simul, location, vehicle, time_horizon, driving_time, w_viol, w_drive, w_dev, w_net, first_station):
    # converting values from new simulator
    vehicle_current_batteries = vehicle.battery_inventory
    incoming_flat_bike_rate_plus_incoming_charged_bike_rate = location.get_arrive_intensity(simul.day(), simul.hour())
    location_incoming_charged_bike_rate = location.get_arrive_intensity(simul.day(), simul.hour())
    demand_per_hour = location.get_leave_intensity(simul.day(), simul.hour())
    vehicle_current_charged_bikes = len(vehicle.scooter_inventory)
    location_current_charged_bikes = len(location.get_available_scooters())
    location_current_flat_bikes = len(location.get_swappable_scooters(settings.BATTERY_LIMIT))
    vehicle_current_station_current_charged_bikes = len(vehicle.current_location.get_available_scooters())
    location_get_incoming_charged_rate = location.get_arrive_intensity(simul.day(), simul.hour())
    location_get_outgoing_customer_rate = location.get_leave_intensity(simul.day(), simul.hour())
    location_available_parking = location.capacity - len(location.scooters)
    vehicle_available_bike_capacity = vehicle.scooter_inventory_capacity - len(vehicle.scooter_inventory)

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
        t_cong = (location.capacity - location_current_charged_bikes - location_current_flat_bikes)/(
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
    if (vehicle_current_station_current_charged_bikes - vehicle.current_location.get_target_state(simul.day(), simul.hour())) > 0 and (incoming_flat_bike_rate_plus_incoming_charged_bike_rate -
             demand_per_hour) > 0 and first_station and location_current_charged_bikes > location.get_target_state(simul.day(), simul.hour()):
        return -10000
    # ------- Deviation at time horizon  -------
    # Starving station
    if demand_per_hour - location_incoming_charged_bike_rate > 0:
        charged_at_t = location_current_charged_bikes - (demand_per_hour -
                location_incoming_charged_bike_rate) * min(time_horizon, time_to_starvation)
        if location.get_target_state(simul.day(), simul.hour()) - charged_at_t > 0 and first_station and (vehicle_current_charged_bikes < 2 and (
                vehicle_current_batteries < 2 or location_current_flat_bikes < 2)):
            return -10000
    # Congesting station
    elif demand_per_hour - location_incoming_charged_bike_rate < 0:
        charged_at_t = location_current_charged_bikes + (location_incoming_charged_bike_rate
                - demand_per_hour) * min(time_horizon, time_to_congestion)
        if location_available_parking == 0 and first_station and vehicle_available_bike_capacity < 2:
            return -10000
    else:
        charged_at_t = location_current_charged_bikes
    dev = abs(location.get_target_state(simul.day(), simul.hour()) - charged_at_t)
    net = abs(location_get_incoming_charged_rate - location_get_outgoing_customer_rate)
    return - w_viol * time_to_violation - w_drive * driving_time + w_dev * dev + w_net * net

def get_station_car_travel_time(state, station, end_st_id):
    return state.get_distance(station.id, end_st_id) / settings.VEHICLE_SPEED

class HeuristicManager:

    time_h = 25

    def __init__(self, simul, vehicles, station_full_set, no_scenarios=1, init_branching=3, weights=None,
                 criticality=True, writer=None, crit_weights=None):
        self.simul = simul
        self.no_scenarios = no_scenarios
        self.customer_arrival_scenarios = list()
        self.vehicles = vehicles
        self.station_set = station_full_set
        self.route_patterns = list()
        self.subproblem_scores = list()
        self.master_solution = None
        self.init_branching = init_branching
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
        gen = GenerateRoutePattern(self.simul, vehicle.current_location, self.station_set, vehicle,
                                   init_branching=self.init_branching, criticality=self.criticality,
                                   crit_weights=self.crit_weights)
        gen.get_columns()
        self.route_patterns.append(gen)
        model_man = ModelManager(vehicle, self.simul)
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
                station_get_incoming_charged_rate = station.get_arrive_intensity(self.simul.day(), self.simul.hour())
                station_get_incoming_flat_rate = station.get_arrive_intensity(self.simul.day(), self.simul.hour())
                station_get_outgoing_customer_rate = station.get_leave_intensity(self.simul.day(), self.simul.hour())

                c1_times = HeuristicManager.poisson_simulation(self.simul.state.rng, station_get_incoming_charged_rate / 60, HeuristicManager.time_h)
                c2_times = HeuristicManager.poisson_simulation(self.simul.state.rng, station_get_incoming_flat_rate / 60, HeuristicManager.time_h)
                c3_times = HeuristicManager.poisson_simulation(self.simul.state.rng, station_get_outgoing_customer_rate / 60, HeuristicManager.time_h)
                scenario.append([c1_times, c2_times, c3_times])
            self.customer_arrival_scenarios.append(scenario)

    @staticmethod
    def poisson_simulation(rng, intensity_rate, time_steps):
        times = list()
        for t in range(time_steps):
            arrival = rng.poisson(intensity_rate)
            for i in range(arrival):
                times.append(t)
        return times
