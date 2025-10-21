from policies.hlv_master.FF_BS_PILOT_policy import BS_PILOT_FF, get_max_num_usable_escooters, calculate_loading_quantities_and_swaps_greedy as calculate_loading_quantities_and_swaps_greedy_ff
from policies.hlv_master.SB_BS_PILOT_policy import calculate_loading_quantities_and_swaps_greedy as calculate_loading_quantities_and_swaps_greedy_sb
from settings import *
import sim
from policies.hlv_master.Visit import Visit
from policies.hlv_master.Plan import Plan
from .collab3_Criticality_Score import calculate_criticality_ff, calculate_criticality_sb
from .FF_Criticality_score import calculate_cluster_type 
from .SB_Criticality_score import calculate_station_type
from policies.hlv_master.Simple_calculations import copy_arr_iter, calculate_net_demand, generate_discounting_factors, calculate_hourly_discharge_rate
from policies.hlv_master.dynamic_clustering import find_clusters
import numpy as np
import time

class Collab3(BS_PILOT_FF):
    def __init__(self, 
                max_depth = MAX_DEPTH, 
                number_of_successors = NUM_SUCCESSORS, 
                time_horizon = TIME_HORIZON, 
                criticality_weights_set = None,
                evaluation_weights = EVALUATION_WEIGHTS, 
                number_of_scenarios = NUM_SCENARIOS, 
                discounting_factor = DISCOUNTING_FACTOR,
                congestion_criteria = CONGESTION_CRITERIA,
                starvation_criteria = STARVATION_CRITERIA,
                swap_threshold = BATTERY_LIMIT_TO_SWAP,
                criticality_weights_set_ff = CRITICAILITY_WEIGHTS_SET_FF,
                criticality_weights_set_sb = CRITICAILITY_WEIGHTS_SET_SB,
                operator_radius = OPERATOR_RADIUS,
                num_clusters = MAX_NUMBER_OF_CLUSTERS,
                adjusting_criticality = ADJUSTING_CRITICALITY
                ):
        super().__init__(
            max_depth = max_depth,
            number_of_successors = number_of_successors, 
            time_horizon = time_horizon, 
            criticality_weights_set = criticality_weights_set, 
            evaluation_weights = evaluation_weights, 
            number_of_scenarios = number_of_scenarios, 
            discounting_factor = discounting_factor,
            swap_threshold = swap_threshold,
            operator_radius = operator_radius,
            num_clusters = num_clusters
        )
        self.criticality_weights_set_ff = criticality_weights_set_ff
        self.criticality_weights_set_sb = criticality_weights_set_sb
        self.adjusting_criticality = adjusting_criticality
        self.congestion_criteria = congestion_criteria
        self.starvation_criteria = starvation_criteria
    
    def get_best_action(self, simul, vehicle):
        """
        Returns an Action (with which escooters to swap batteries on, which bikes to pick-up, which bikes to unload, next location to drive to)
        Additionally, it returns the cluster for which the next location is a part of.

        Parameters:
        - simul = simulator
        - vehicle = Vehicle-object that is doing the action
        """
        start_logging_time = time.time() 
        next_location = None
        bikes_to_pickup = []
        bikes_to_deliver = []
        batteries_to_swap = []

        end_time = simul.time + self.time_horizon 
        total_num_sb_bikes_in_system = len(simul.state.get_all_sb_bikes())
        total_num_ff_bikes_in_system = len(simul.state.get_all_ff_bikes())

        # Loading and swap strategy at current area or cluster is always chosen greedily
        if isinstance(vehicle.location, sim.Depot): # No action needed if at depot
            bikes_to_pickup = []
            bikes_to_deliver = []
            batteries_to_swap = []
            num_bikes_pickup = len(bikes_to_pickup)
            num_bikes_deliver = len(bikes_to_deliver)
            num_batteries_to_swap = len(batteries_to_swap)
        elif vehicle.cluster is None:
            bikes_to_pickup, bikes_to_deliver, batteries_to_swap = calculate_loading_quantities_and_swaps_greedy(vehicle, simul, vehicle.location, self.congestion_criteria, self.starvation_criteria, self.swap_threshold)
            num_bikes_pickup = len(bikes_to_pickup)
            num_bikes_deliver = len(bikes_to_deliver)
            num_batteries_to_swap = len(batteries_to_swap)

            simul.metrics.add_aggregate_metric(simul, 'num battery swaps', num_batteries_to_swap)
            simul.metrics.add_aggregate_metric(simul, 'num bike pickups', num_bikes_pickup)
            simul.metrics.add_aggregate_metric(simul, 'num bike deliveries', num_bikes_deliver)
        else:
            bikes_to_pickup, bikes_to_deliver, batteries_to_swap = calculate_loading_quantities_and_swaps_greedy(vehicle, simul, vehicle.cluster, self.congestion_criteria, self.starvation_criteria, self.swap_threshold)
            num_bikes_pickup = len(bikes_to_pickup)
            num_bikes_deliver = len(bikes_to_deliver)
            num_batteries_to_swap = len(batteries_to_swap)

            simul.metrics.add_aggregate_metric(simul, 'num battery swaps', num_batteries_to_swap)
            simul.metrics.add_aggregate_metric(simul, 'num escooter pickups', num_bikes_pickup)
            simul.metrics.add_aggregate_metric(simul, 'num escooter deliveries', num_bikes_deliver)

        # Goes to depot if the vehicle's battery inventory is empty on arrival, and picks up all escooters at location that is unusable
        if vehicle.battery_inventory <= 0 and len(simul.state.depots) > 0 and num_bikes_pickup + num_batteries_to_swap > 0:
            next_location = simul.state.get_closest_depot(vehicle)
            # If no depot, just stay and do nothing
            if next_location == vehicle.location.location_id:
                return sim.Action(
                [],
                [],
                [],
                next_location
            )
            bikes_to_pickup = [bike.bike_id for bike in vehicle.location.get_swappable_bikes(self.swap_threshold)]
            max_pickup = min(vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()), len(bikes_to_pickup))
            return sim.Action(
                [],
                bikes_to_pickup[:max_pickup],
                [],
                next_location
            )

        middle_logging_time = time.time() 
        simul.metrics.add_aggregate_metric(simul, "accumulated find action time", middle_logging_time - start_logging_time)

        # Make a plan for all vehicles
        # Dictionary with key = vehicle_id, 
        #   value = List of Visits, where the first one is the current location of vehicle, or the location the vehicle is on its way to
        plan_dict = dict()
        for v in simul.state.get_vehicles():
            # If vehicle is at a location, add current location to the plan with the greedy loading and swap strategy
            if v.eta == 0:
                plan_dict[v.vehicle_id] = [Visit(v.location, num_bikes_pickup, num_bikes_deliver, num_batteries_to_swap, simul.time, v)]
            
            # If the vehicle is driving, use pilot to calculate the loading and swap strategy and add to the plan
            else:
                num_bikes_pickup, num_bikes_deliver, num_batteries_to_swap = self.calculate_loading_quantities_and_swaps_pilot(len(v.get_sb_bike_inventory()), len(v.get_ff_bike_inventory()), v.battery_inventory, simul, v.location, v.eta)
                plan_dict[v.vehicle_id] = [Visit(v.location, int(num_bikes_pickup), int(num_bikes_deliver), int(num_batteries_to_swap), v.eta, v)]
        
        # All locations the vehicles are at or are on their way to is added to the tabu list and plan
        tabu_list = [v.location.location_id for v in simul.state.get_vehicles()]
        plan = Plan(plan_dict, tabu_list)

        # Use X-PILOT-BS to find which location to drive to next
        next_location, cluster = self.PILOT_function(simul, 
                                                     vehicle,
                                                     plan, 
                                                     self.max_depth, 
                                                     self.number_of_successors, 
                                                     end_time, 
                                                     total_num_sb_bikes_in_system,
                                                     total_num_ff_bikes_in_system)
        
        end_time = time.time()
        
        simul.metrics.add_aggregate_metric(simul, "visited cluster", 1) if cluster else simul.metrics.add_aggregate_metric(simul, "visited stations", 1)
        simul.metrics.add_aggregate_metric(simul, "accumulated find location time", end_time - middle_logging_time)
        simul.metrics.add_aggregate_metric(simul, "accumulated sol time", end_time - start_logging_time)        
        simul.metrics.add_aggregate_metric(simul, 'get_best_action', 1)
        
        return sim.Action(
            batteries_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_location,
            cluster
        )

    def PILOT_function(self, simul, vehicle, initial_plan, max_depth, num_successors, end_time, total_num_sb_bikes_in_system, total_num_ff_bikes_in_system):
        hourly_discharge = calculate_hourly_discharge_rate(simul, total_num_ff_bikes_in_system, True, False)
        completed_plans = []
        for i in range(len(self.criticality_weights_set_ff)):
            plans = [[] for i in range(max_depth +1)]
            plans[0].append(initial_plan)
            weight_set = [self.criticality_weights_set_ff[i], self.criticality_weights_set_sb[i]]

            for depth in range(1, max_depth+1):
                if depth > 1:
                    num_successors = max(1, round(num_successors/2))
                
                while plans[depth-1] != []:
                    plan = plans[depth-1].pop(0)
                    next_vehicle = plan.next_visit.vehicle

                    if next_vehicle != vehicle:
                        num_successors_other_veheicle = max(1, round(num_successors/2))
                        new_visits = self.greedy_next_visit(plan, simul, num_successors_other_veheicle, weight_set, total_num_sb_bikes_in_system, total_num_ff_bikes_in_system, hourly_discharge)
                    else:
                        new_visits = self.greedy_next_visit(plan, simul, num_successors, weight_set, total_num_sb_bikes_in_system, total_num_ff_bikes_in_system, hourly_discharge)
                    
                    if new_visits == None or plan.next_visit.get_depature_time() > end_time:
                        new_plan = Plan(plan.copy_plan(),copy_arr_iter(plan.tabu_list), weight_set, plan.branch_number)
                        plans[depth].append(new_plan)
                    else:
                        for branch_number, visit in enumerate(new_visits):
                            new_plan_dict = plan.copy_plan()
                            new_plan_dict[next_vehicle.vehicle_id].append(visit)
                            tabu_list = copy_arr_iter(plan.tabu_list)
                            tabu_list.append(visit.station.location_id)

                            if depth == 1:
                                new_plan = Plan(new_plan_dict, tabu_list, weight_set, branch_number)
                            else:
                                new_plan = Plan(new_plan_dict, tabu_list, weight_set, plan.branch_number)
                            

                            if next_vehicle.vehicle_id == vehicle.vehicle_id:
                                plans[depth].append(new_plan)
                            else:
                                plans[depth-1].append(new_plan)


            for plan in plans[max_depth]:

                dep_time = plan.next_visit.get_depature_time()
                temp_plan = Plan(plan.copy_plan(), copy_arr_iter(plan.tabu_list), weight_set, plan.branch_number)

                if dep_time > end_time and temp_plan.branch_number == None:
                    # print(f'dep_time({dep_time}) > end_time({end_time}), when time_horizon={self.time_horizon}')
                    dep_time = end_time - 1

                while dep_time < end_time:
                    new_visit = self.greedy_next_visit(temp_plan, simul, 1, weight_set, total_num_sb_bikes_in_system, total_num_ff_bikes_in_system, hourly_discharge)
                    
                    if new_visit != None:
                        new_visit = new_visit[0]
                        temp_plan.tabu_list.append(new_visit.station.location_id)
                        temp_plan.plan[temp_plan.next_visit.vehicle.vehicle_id].append(new_visit)
                        dep_time = new_visit.get_depature_time()
                        temp_plan.find_next_visit()
                    else:
                        break

                    if dep_time < 0:
                        print("Dep tid er negetiv?", dep_time, new_visit)

                completed_plans.append(temp_plan)
            
        plan_scores = dict()
        scenarios = self.generate_scenarioes(simul, self.number_of_scenarios, poisson = True)
        for plan in completed_plans:
            plan_scores[plan] = []
            for scenario_dict in scenarios:
                score = 0
                for v in plan.plan:
                    score += self.evaluate_route(plan.plan[v], scenario_dict, end_time, simul, self.evaluation_weights, hourly_discharge)
                plan_scores[plan].append(score)
        
        # Returns the station with the best avarage score over all scenarios
        return self.return_best_move_average(vehicle, simul, plan_scores)

    def calculate_loading_quantities_and_swaps_pilot(self, num_sb_bikes_now, num_ff_bikes_now, battery_inventory_now, simul, location, eta):
        """
        Returns the NUMBER of bikes to pick up, deliver and swap batteries on. Takes future demand into account by calculating it for the next hour, and treats it as evenly distributed throughout that hour

        Parameters:
        - vehicle = Vehicle-object the action is considered for
        - simul = Simulator
        - cluster = cluster the vehicle is considering doing the action at
        - eta = Estimated time of arrival for the vehicle to arrive at cluster
        """
        if isinstance(location, sim.Station):
            target_state = round(location.get_target_state(simul.day(),simul.hour())) 
            time_until_arrival = eta - simul.time

            # Calculate number of bikes arriving/leaving in the time period until vehicle arrival
            net_demand = calculate_net_demand(location, simul.time, simul.day(), simul.hour(), min(time_until_arrival, 60))
            max_num_usable_bikes = len(location.get_available_bikes()) + min(len(location.get_unusable_bikes()), battery_inventory_now)
            max_num_usable_bikes_eta = max_num_usable_bikes + net_demand

            # Calculate the amount of neighbors that are starving or congested, can impact the number of bikes to operate on
            num_starved_neighbors = 0
            num_congested_neighbors = 0

            for neighbor in location.get_neighbours():
                net_demand_neighbor = calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(), min(time_until_arrival, 60))
                num_usable_bikes_neighbor_eta = len(neighbor.get_available_bikes()) + net_demand_neighbor
                if num_usable_bikes_neighbor_eta < self.starvation_criteria * neighbor.capacity:
                    num_starved_neighbors += 1
                elif num_usable_bikes_neighbor_eta > self.congestion_criteria * neighbor.capacity:
                    num_congested_neighbors += 1

            number_of_bikes_pickup = 0
            number_of_bikes_deliver = 0
            number_of_battery_swaps = 0

            # Calculate how many bikes to do different actions on
            if max_num_usable_bikes_eta < target_state: # deliver
                number_of_bikes_deliver = min(num_sb_bikes_now, 
                                            target_state - max_num_usable_bikes_eta + BIKES_STARVED_NEIGHBOR * num_starved_neighbors)
                number_of_battery_swaps = min(len(location.get_swappable_bikes(self.swap_threshold)), 
                                            battery_inventory_now)
            
            elif max_num_usable_bikes_eta > target_state: # pick-up
                remaining_cap_vehicle = VEHICLE_BATTERY_INVENTORY - num_sb_bikes_now
                num_less_bikes = min(remaining_cap_vehicle, 
                                    max_num_usable_bikes_eta - target_state + BIKES_CONGESTED_NEIGHBOR * num_congested_neighbors)

                num_swappable_bikes = len(location.get_swappable_bikes(self.swap_threshold))
                vehicle_battery_inventory = battery_inventory_now

                num_bikes_only_swap = min(max(0, target_state - len(location.get_available_bikes())), 
                                                vehicle_battery_inventory)
                vehicle_battery_inventory -= num_bikes_only_swap
                
                # Restriction on how many bikes can be swapped and picked up
                num_bikes_to_swap_and_pickup = min( num_swappable_bikes,
                                                    vehicle_battery_inventory, 
                                                    num_less_bikes)

                # Update the number of batteries left in vehicle
                vehicle_battery_inventory -= num_bikes_to_swap_and_pickup
                
                # Number of escooters that can only be picked up, not swapped battery on
                num_escooters_to_only_pickup = num_less_bikes - num_bikes_to_swap_and_pickup

                # Number of escooters that can only have their battery swapped, but stays at the location
                num_bikes_only_swap += min(vehicle_battery_inventory,
                                                max(0, num_swappable_bikes - num_bikes_to_swap_and_pickup - num_bikes_only_swap))
                
                number_of_bikes_pickup = num_bikes_to_swap_and_pickup + num_escooters_to_only_pickup
                number_of_battery_swaps = num_bikes_only_swap
            
            else: # only swap
                swappable_bikes = location.get_swappable_bikes(self.swap_threshold)
                number_of_battery_swaps = min(len(swappable_bikes), battery_inventory_now)
            
            return number_of_bikes_pickup, number_of_bikes_deliver, number_of_battery_swaps
        else:
            return super().calculate_loading_quantities_and_swaps_pilot(num_ff_bikes_now, battery_inventory_now, simul, location, eta)
        
    def greedy_next_visit(self, plan, simul, number_of_successors, weight_set, total_num_sb_bikes_in_system, total_num_ff_bikes_in_system, hourly_discharge):
        visits = []
        tabu_list = plan.tabu_list
        vehicle = plan.next_visit.vehicle

        num_sb_bikes_now = len(vehicle.get_sb_bike_inventory())
        num_ff_bikes_now = len(vehicle.get_ff_bike_inventory())
        num_battery_inventory_now = vehicle.battery_inventory

        # Update the vehicle bike inventory based on the planned operational actions
        for visit in plan.plan[vehicle.vehicle_id]:
            if isinstance(visit.station, sim.Station):
                num_sb_bikes_now += visit.loading_quantity
                num_sb_bikes_now -= visit.unloading_quantity
                num_battery_inventory_now -= visit.swap_quantity
            else:
                num_ff_bikes_now += visit.loading_quantity
                num_ff_bikes_now -= visit.unloading_quantity
                num_battery_inventory_now -= visit.swap_quantity

        # Finds potential next stations based on pick up or delivery status of the station and tabulist
        potential_stations = find_potential_stations(simul,LOCATION_TYPE_MARGIN, LOCATION_TYPE_MARGIN,vehicle, num_sb_bikes_now, num_ff_bikes_now+num_sb_bikes_now, tabu_list)
        time_of_departure = plan.plan[vehicle.vehicle_id][-1].get_depature_time()
        potential_clusters = find_potential_clusters(simul, LOCATION_TYPE_MARGIN, vehicle, num_ff_bikes_now, num_sb_bikes_now, num_battery_inventory_now, time_of_departure, plan.plan[vehicle.vehicle_id][-1].station, self.operator_radius, self.num_clusters)
        if potential_stations == [] and potential_clusters == []:
            return None
        
        number_of_successors = min(number_of_successors, len(potential_stations)+len(potential_clusters))

        # Finds the criticality score of all potential locations, and sort them in descending order
        stations_sorted = calculate_criticality_sb(weight_set[1], simul, potential_stations, plan.plan[vehicle.vehicle_id][-1].station, total_num_sb_bikes_in_system, tabu_list) if potential_stations != [] else {}
        clusters_sorted = calculate_criticality_ff(weight_set[0], simul, potential_clusters, plan.plan[vehicle.vehicle_id][-1].station, total_num_ff_bikes_in_system, tabu_list, self.adjusting_criticality, hourly_discharge) if potential_clusters != [] else {}
        sorted_criticalities_merged = dict(sorted({**stations_sorted, **clusters_sorted}.items(), key=lambda item: item[1], reverse=True))
        destinations_sorted_list = list(sorted_criticalities_merged.keys())
        
        # Selects the most critical clusters as next visits
        next_destination = [destinations_sorted_list[i] for i in range(number_of_successors)]

        for next in next_destination:
            arrival_time = plan.plan[vehicle.vehicle_id][-1].get_depature_time() + simul.state.get_vehicle_travel_time(plan.plan[vehicle.vehicle_id][-1].station.location_id, next.location_id) + MINUTES_CONSTANT_PER_ACTION
            number_of_escooters_to_pickup, number_of_escooters_to_deliver, number_of_escooters_to_swap = self.calculate_loading_quantities_and_swaps_pilot(num_sb_bikes_now, num_ff_bikes_now, num_battery_inventory_now, simul, next, arrival_time)
            new_visit = Visit(next, number_of_escooters_to_pickup, number_of_escooters_to_deliver, number_of_escooters_to_swap, arrival_time, vehicle)
            visits.append(new_visit)
        
        return visits
    
    def evaluate_route(self, route, scenario_dict, end_time, simul, weights, hourly_discharge):
        """
        Returns the score based on if the vehicle drives this route in comparisson to not driving it at all

        Parameters:
        - route = list of visits the vehicle is supposed to do
        - scenario_dict = a dictionary with a possible net demand for each area
        - end_time = the stopping point of the horizon to evaluate
        - simul = Simulator
        - weights = weights for avoided violations, neighbor roamings, and improved deviation
        - total_num_bikes_in_system = the total amount of bicycles that are in the SB system
        """

        discounting_factors = generate_discounting_factors(len(route), self.discounting_factor)
        avoided_disutility = 0
        current_time = simul.time
        counter = 0 # which stage during the visit the vehicle is at -> used to discount the score

        for visit in route:
            avoided_violations = 0
            neighbor_roamings = 0
            improved_deviation = 0

            loading_quantity = visit.loading_quantity
            unloading_quantity = visit.unloading_quantity
            swap_quantity = visit.swap_quantity

            location = visit.station
            if isinstance(location, sim.Station):
                hourly_discharge = 0
                neighbors = location.get_neighbours()

                eta = visit.arrival_time

                if eta > end_time:
                    eta = end_time
                
                initial_inventory = len(location.get_available_bikes()) # TODO congestion behandles annerledes
                net_demand = scenario_dict[location.location_id]
                target_state = location.get_target_state(simul.day(), simul.hour())

                # Calculate when the first starvation or congestion will occur if not visited
                if net_demand < 0:
                    sorted_bikes_in_station = sorted(location.get_bikes(), key=lambda bike: bike.battery, reverse=False)

                    # Calculate hours until violation because no bikes have sufficient battery
                    battery_top3 = [Ebike.battery for Ebike in sorted_bikes_in_station[-3:]]
                    average_battery_top3 = sum(battery_top3)/len(battery_top3) if battery_top3 != [] else 0
                    hours_until_violation_battery = average_battery_top3/hourly_discharge if hourly_discharge != 0 else 8

                    # Find the earlist moment for a violation
                    hours_until_first_violation = min(
                                                    (len(location.get_available_bikes())/ -net_demand), # How long until the net demand results in a starvation
                                                    hours_until_violation_battery
                                                    )
                    
                    # Find the time in minutes for the violation
                    time_of_first_violation_no_visit = current_time + (hours_until_first_violation * 60)
                    
                elif net_demand > 0:
                    # How long until the net demand results in a congestion
                    hours_until_first_violation = (location.capacity - location.number_of_bikes()) / net_demand
                    time_of_first_violation_no_visit = current_time + (hours_until_first_violation * 60)
                else:
                    time_of_first_violation_no_visit = end_time
                
                # Calculate number of violation within the time horizon
                if end_time > time_of_first_violation_no_visit:
                    num_violation_no_visit = ((end_time - time_of_first_violation_no_visit)/60) * abs(net_demand)
                else:
                    num_violation_no_visit = 0
                
                # Violations that we can't avoid due to driving time
                if eta > time_of_first_violation_no_visit:
                    unavoidable_violations = ((eta - time_of_first_violation_no_visit)/60) * abs(net_demand)
                else:
                    unavoidable_violations = 0
                
                # Number of bikes at station after visit
                station_inventory_after_visit = initial_inventory + ((eta - current_time)/60) * abs(net_demand) - unavoidable_violations - loading_quantity + unloading_quantity + swap_quantity
                
                # Time for first violation if we visit the station
                if net_demand < 0:
                    time_until_first_violation = (station_inventory_after_visit / (-net_demand)) * 60
                    if swap_quantity > loading_quantity + 3: # Knowing top 3 bikes at station are fully charged
                        time_first_violation_after_visit = eta + min(time_until_first_violation, 100/(hourly_discharge if hourly_discharge != 0 else 1) * 60)
                    else:
                        time_first_violation_after_visit = eta + min(time_until_first_violation, (average_battery_top3)/(hourly_discharge if hourly_discharge != 0 else 1) * 60)
                elif net_demand > 0:
                    time_until_first_violation = (location.capacity - station_inventory_after_visit) / net_demand
                    time_first_violation_after_visit = eta + time_until_first_violation * 60
                else:
                    time_first_violation_after_visit = end_time
                
                if time_first_violation_after_visit < end_time:
                    violations_after_visit = ((end_time - time_first_violation_after_visit)/60) * abs(net_demand)
                else:
                    violations_after_visit = 0

                # How many violations did we manage to avoid, not counting the ones we could not do anything about
                avoided_violations = num_violation_no_visit - violations_after_visit - unavoidable_violations

                # Calculating the deviation from target at end time
                if net_demand <= 0:
                    ending_inventory_after_visit = max(0, station_inventory_after_visit + ((end_time - eta)/60) * net_demand)
                    ending_inventory_no_visit = max(0, initial_inventory + ((end_time - current_time)/60) * net_demand)
                else:
                    ending_inventory_after_visit = min(location.capacity, station_inventory_after_visit + ((end_time-eta)/60)*net_demand)
                    ending_inventory_no_visit = min(location.capacity , initial_inventory + ((end_time-current_time)/60)*net_demand)

                deviation_after_visit = abs(ending_inventory_after_visit - target_state)
                deviation_no_visit = abs(ending_inventory_no_visit - target_state)

                improved_deviation = deviation_no_visit - deviation_after_visit

                # Calculate excess bikes and locks, with and wihtout visits
                excess_bikes_after_visit = ending_inventory_after_visit
                excess_locks_after_visit = location.capacity - ending_inventory_after_visit
                if net_demand > 0:
                    excess_bikes_no_visit = min(location.capacity, initial_inventory + ((end_time-current_time)/60) * net_demand)
                    excess_locks_no_visit = max(0, location.capacity - (initial_inventory + ((end_time-current_time)/60) * net_demand))
                elif net_demand <= 0:
                    excess_bikes_no_visit = max(0, initial_inventory + ((end_time-current_time)/60) * net_demand)
                    excess_locks_no_visit = min(location.capacity, location.capacity - (initial_inventory+((end_time-current_time)/60) * net_demand))
                
                # Calculate station type
                station_type = calculate_station_type(target_state, station_inventory_after_visit)

                for neighbor in neighbors:
                    roamings = 0
                    roamings_no_visit = 0
                    net_demand_neighbor = scenario_dict[neighbor.location_id]
                    expected_bikes_neighbor = len(neighbor.get_available_bikes()) + net_demand_neighbor
                    neighbor_type = calculate_station_type(neighbor.get_target_state(simul.day(),simul.hour()), expected_bikes_neighbor)

                    if neighbor_type == station_type:
                        if net_demand_neighbor < 0:
                            time_first_violation = current_time + (len(neighbor.get_available_bikes())/ (-net_demand_neighbor)) * 60
                        elif net_demand_neighbor > 0:
                            time_first_violation = current_time + ((neighbor.capacity - neighbor.number_of_bikes()) / (net_demand_neighbor)) * 60
                        else:
                            time_first_violation = end_time
                        

                        if time_first_violation < end_time:
                            convertable_violations = (min(end_time - time_first_violation, end_time - eta)/60) * abs(net_demand_neighbor)

                            # Count the roamings done with and without visitation
                            if neighbor_type == 'p':
                                if convertable_violations <= excess_locks_after_visit:
                                    roamings += convertable_violations
                                    excess_locks_after_visit -= convertable_violations
                                else:
                                    roamings += excess_locks_after_visit
                                    excess_locks_after_visit -= excess_locks_after_visit
                                
                                if convertable_violations <= excess_locks_no_visit:
                                    roamings_no_visit += convertable_violations 
                                    excess_locks_no_visit -= convertable_violations
                                else:
                                    roamings_no_visit += excess_locks_no_visit
                                    excess_locks_no_visit -= excess_locks_no_visit

                            if neighbor_type == 'd':
                                if convertable_violations <= excess_bikes_after_visit:
                                    roamings += convertable_violations
                                    excess_bikes_after_visit -= convertable_violations
                                else:
                                    roamings += excess_bikes_after_visit
                                    excess_bikes_after_visit -= excess_bikes_after_visit
                                
                                if convertable_violations <= excess_bikes_no_visit:
                                    roamings_no_visit += convertable_violations
                                    excess_bikes_no_visit -= convertable_violations
                                else:
                                    roamings_no_visit += excess_bikes_no_visit
                                    excess_bikes_no_visit -= excess_bikes_no_visit

                    distance_scaling = ((simul.state.get_vehicle_travel_time(location.location_id, neighbor.location_id)/60)*VEHICLE_SPEED)/MAX_ROAMING_DISTANCE_SOLUTIONS
                    neighbor_roamings += (1 - distance_scaling) * (roamings - roamings_no_visit)
                
                avoided_disutility += discounting_factors[counter]*(weights[0]*avoided_violations + weights[1]*neighbor_roamings + weights[2]*improved_deviation)

                counter += 1
            else:
                neighbors = location.get_neighbours()

                eta = visit.arrival_time

                if eta > end_time:
                    eta = end_time
                
                initial_inventory = len(location.get_available_bikes())
                net_demand = scenario_dict[location.location_id]
                target_state = location.get_target_state(simul.day(), simul.hour())

                # Calculate when the first starvation or congestion will occur if not visited
                if net_demand < 0:
                    sorted_escooters_at_area = sorted(location.get_bikes(), key=lambda bike: bike.battery, reverse=False)

                    # Calculate hours until violation because no bikes have sufficient battery
                    battery_top3 = [Ebike.battery for Ebike in sorted_escooters_at_area[-3:]]
                    average_battery_top3 = sum(battery_top3)/len(battery_top3) if battery_top3 != [] else 0
                    hours_until_violation_battery = average_battery_top3/hourly_discharge if hourly_discharge != 0 else 8 

                    # Find the earlist moment for a violation
                    hours_until_first_violation = min(
                                                    (len(location.get_available_bikes())/ -net_demand), # How long until the net demand results in a starvation
                                                    hours_until_violation_battery
                                                    )
                    
                    # Find the time in minutes for the violation
                    time_of_first_violation_no_visit = current_time + (hours_until_first_violation * 60)
                else:
                    time_of_first_violation_no_visit = end_time
                
                # Calculate number of violation within the time horizon
                if end_time > time_of_first_violation_no_visit:
                    num_violation_no_visit = ((end_time - time_of_first_violation_no_visit)/60) * abs(net_demand)
                else:
                    num_violation_no_visit = 0
                
                # Violations that we can't avoid due to driving time
                if eta > time_of_first_violation_no_visit:
                    unavoidable_violations = ((eta - time_of_first_violation_no_visit)/60) * abs(net_demand)
                else:
                    unavoidable_violations = 0
                
                # Number of bikes at station after visit
                area_inventory_after_visit = initial_inventory + ((eta - current_time)/60) * abs(net_demand) - unavoidable_violations - loading_quantity + unloading_quantity + swap_quantity
                
                # Time for first violation if we visit
                if net_demand < 0:
                    time_until_first_violation = (area_inventory_after_visit / (-net_demand)) * 60
                    if swap_quantity > loading_quantity + 3: # Knowing top 3 bikes at station are fully charged
                        time_first_violation_after_visit = eta + min(time_until_first_violation, 100/hourly_discharge * 60 if hourly_discharge != 0 else 480)
                    else:
                        time_first_violation_after_visit = eta + min(time_until_first_violation, (average_battery_top3)/(hourly_discharge) * 60 if hourly_discharge != 0 else 480)
                else:
                    time_first_violation_after_visit = end_time
                
                if time_first_violation_after_visit < end_time:
                    violations_after_visit = ((end_time - time_first_violation_after_visit)/60) * net_demand
                else:
                    violations_after_visit = 0

                # How many violations did we manage to avoid, not counting the ones we could not do anything about
                avoided_violations = num_violation_no_visit - violations_after_visit - unavoidable_violations

                ending_inventory_after_visit = max(0, area_inventory_after_visit + ((end_time - eta)/60) * net_demand)
                deviation_visit = abs(ending_inventory_after_visit - target_state)
    
                ending_inventory_no_visit = max(0, initial_inventory + ((end_time - current_time)/60) * net_demand)
                deviation_no_visit = abs(ending_inventory_no_visit - target_state)

                improved_deviation = deviation_no_visit - deviation_visit

                excess_escooters = ending_inventory_after_visit
                excess_escooters_no_visit = ending_inventory_no_visit

                expected_number_of_escooters = area_inventory_after_visit
                area_type = calculate_cluster_type(target_state, expected_number_of_escooters)

                for neighbor in neighbors:
                    roamings = 0
                    roamings_no_visit = 0
                    net_demand_neighbor = scenario_dict[neighbor.location_id]
                    expected_ecooters_neighbor = len(neighbor.get_available_bikes()) + net_demand_neighbor
                    neighbor_type = calculate_cluster_type(neighbor.get_target_state(simul.day(),simul.hour()),expected_ecooters_neighbor)

                    if neighbor_type == area_type:
                        if net_demand_neighbor < 0:
                            time_first_violation = current_time + (len(neighbor.get_available_bikes())/-net_demand_neighbor) * 60
                        else:
                            time_first_violation = end_time
                        

                        if time_first_violation < end_time:
                            convertable_violations = (min(end_time - time_first_violation, end_time - eta)/60) * net_demand_neighbor

                            if neighbor_type == 'd':
                                if abs(convertable_violations) <= excess_escooters:
                                    roamings += abs(convertable_violations)
                                    excess_escooters -= abs(convertable_violations)
                                else:
                                    roamings += excess_escooters
                                    excess_escooters -= excess_escooters
                                
                                if abs(convertable_violations) <= excess_escooters_no_visit:
                                    roamings_no_visit += abs(convertable_violations)
                                    excess_escooters_no_visit -= abs(convertable_violations)
                                else:
                                    roamings_no_visit += excess_escooters_no_visit
                                    excess_escooters_no_visit -= excess_escooters_no_visit
                            
                
                    distance_scaling = ((simul.state.get_vehicle_travel_time(location.location_id, neighbor.location_id)/60)* VEHICLE_SPEED)/MAX_ROAMING_DISTANCE_SOLUTIONS
                    neighbor_roamings += (1-distance_scaling)*(roamings-roamings_no_visit)
                
                avoided_disutility += discounting_factors[counter]*(weights[0]*avoided_violations + weights[1]*neighbor_roamings + weights[2]*improved_deviation)

                counter += 1
            
        return avoided_disutility
    
    def return_best_move_average(self, vehicle, simul, plan_scores):
        destination_id, destination = super().return_best_move_average(vehicle, simul, plan_scores)
        if isinstance(destination, sim.Station):
            return destination_id, None
        return destination_id, destination

def calculate_loading_quantities_and_swaps_greedy(vehicle, simul, location, congestion_criteria, starvation_criteria, swap_threshold):
    """
    Returns a list of IDs of the bikes to deliver, pickup or swap batteries on.
    The calculation is done when a vehicle arrives at the station, and the list returned are performed.

    Parameters:
    - vehicle = Vehicle-object that is doing the action
    - simul = Simulator
    - location = The Area or Cluster under consideration
    - congestion_criteria = percentage of station capacity for a station to be considered congested
    - starvation_critera = percentage of station capacity for a station to be considered starved
    """
    if isinstance(location, sim.Station):
        return calculate_loading_quantities_and_swaps_greedy_sb(vehicle, simul, location, congestion_criteria, starvation_criteria, swap_threshold)
    else:
        return calculate_loading_quantities_and_swaps_greedy_ff(vehicle, simul, location, swap_threshold)

def find_potential_clusters(simul, cutoff_vehicle, vehicle, ff_bikes_in_vehicle, sb_bikes_in_vehicle, batteries_in_vehicle, time_of_departure, departure_location, operator_radius, num_clusters):
    """
    Returns a list of Station-Objects that are not in the tabu list, and that need help to reach target state.

    Parameters:
    - simul = Simulator
    - cutoff_vehicle = At what percentage of the vehicle's capacity is the vehicle considered "empty" or "full"
    - vehicle = Vehicle-object under consideration
    - bikes_at_vehicle = number of bikes in the vehicle at the time
    """
    potential_pickup_clusters = []
    potential_delivery_clusters = []

    # Vehicle's inventory is capable of both pickups and deliveries
    if cutoff_vehicle * vehicle.bike_inventory_capacity <= ff_bikes_in_vehicle+sb_bikes_in_vehicle <= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
        max_clusters = num_clusters
        if ff_bikes_in_vehicle >= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
            max_clusters = max_clusters // 2
            potential_delivery_clusters = find_clusters(areas=simul.state.get_areas(), 
                                                  n=max_clusters, 
                                                  max_length=operator_radius, 
                                                  battery_inventory=batteries_in_vehicle, 
                                                  time_now= time_of_departure, 
                                                  departure_location = departure_location,
                                                  operation="delivery",
                                                  simul=simul)
        potential_pickup_clusters = find_clusters(areas=simul.state.get_areas(), 
                                                  n=max_clusters, 
                                                  max_length=operator_radius, 
                                                  battery_inventory=batteries_in_vehicle, 
                                                  time_now= time_of_departure, 
                                                  departure_location = departure_location,
                                                  operation="pickup",
                                                  simul=simul)
        potential_stations = potential_pickup_clusters + potential_delivery_clusters
    else:
        potential_stations = []
        # Vehicle's inventory is not able to deliver escooters
        if ff_bikes_in_vehicle+sb_bikes_in_vehicle <= cutoff_vehicle*vehicle.bike_inventory_capacity:
            potential_stations = find_clusters(areas=simul.state.get_areas(), 
                                                  n=num_clusters, 
                                                  max_length=operator_radius, 
                                                  battery_inventory=batteries_in_vehicle, 
                                                  time_now= time_of_departure, 
                                                  departure_location = departure_location,
                                                  operation="pickup",
                                                  simul=simul)
        # Vehicle's inventory is not able to pick up more escooters
        elif ff_bikes_in_vehicle >= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
            potential_stations = find_clusters(areas=simul.state.get_areas(), 
                                                  n=num_clusters, 
                                                  max_length=operator_radius, 
                                                  battery_inventory=batteries_in_vehicle, 
                                                  time_now= time_of_departure, 
                                                  departure_location = departure_location,
                                                  operation="delivery",
                                                  simul=simul)
    return potential_stations

def find_potential_stations(simul, cutoff_vehicle, cutoff_station, vehicle, sb_bikes_in_vehicle, bikes_in_vehicle, tabu_list):
    """
    Returns a list of Station-Objects that are not in the tabu list, and that need help to reach target state.

    Parameters:
    - simul = Simulator
    - cutoff_vehicle = At what percentage is the vehicle considered "empty" or "full"
    - cutoff_station = At what percentage above or below the target state is a station considered a pickup or delivery
    - vehicle = Vehicle-object under consideration
    - bikes_at_vehicle = number of bikes in the vehicle at the time
    - tabu_list = list of Station-objects that are currently being visited or soon to be
    """
    # Filter out stations in tabulist
    potential_stations = [station for station in simul.state.get_stations() if station.location_id not in tabu_list]
    
    # Makes dictionary with net_demand and target state for all possible stations
    net_demands = {station.location_id: calculate_net_demand(station, simul.time, simul.day(), simul.hour(), 60) for station in potential_stations}
    target_states = {station.location_id: station.get_target_state(simul.day(), simul.hour()) for station in potential_stations}
    
    # If the available bikes in the future is bigger that a cutoff percentage of target state, the station is a pickup station
    potential_pickup_stations = [station for station in potential_stations
                                 if station.number_of_bikes() + net_demands[station.location_id] > (1 + cutoff_station) * target_states[station.location_id]
                                 ] 
    
    # If the available bikes after a visit in the future is lower that a cutoff percentage of target state, the station is a delivery station
    potential_delivery_stations = [ station for station in potential_stations
                                 if station.number_of_bikes() + net_demands[station.location_id] < (1 - cutoff_station) * target_states[station.location_id]
                                 ]
    
    #Decides whether pickup, delivery or balanced is relevant
    if cutoff_vehicle * vehicle.bike_inventory_capacity <= bikes_in_vehicle <= (1 - cutoff_vehicle) * vehicle.bike_inventory_capacity:
        potential_stations = []
        if sb_bikes_in_vehicle >= (1 - cutoff_vehicle) * vehicle.bike_inventory_capacity:
            potential_stations = potential_delivery_stations
        potential_stations += potential_pickup_stations
    else:
        if bikes_in_vehicle <= cutoff_vehicle * vehicle.bike_inventory_capacity:
            potential_stations = potential_pickup_stations

        elif sb_bikes_in_vehicle >= (1 - cutoff_vehicle) * vehicle.bike_inventory_capacity:
            potential_stations = potential_delivery_stations
    
    return potential_stations