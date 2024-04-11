from policies import Policy
from settings import *
import sim
from .Visit import Visit
from .Plan import Plan
from .FF_Criticality_score import calculate_criticality, calculate_cluster_type
from .Simple_calculations import calculate_net_demand, copy_arr_iter, generate_discounting_factors, calculate_hourly_discharge_rate
from .dynamic_clustering import clusterPickup, clusterDelivery

import numpy as np
import time

class BS_PILOT_FF(Policy):
    """
    Class for X-PILOT-BS policy. Finds the best action for e-scooter systems.
    """
    def __init__(self, 
                max_depth = MAX_DEPTH, 
                number_of_successors = NUM_SUCCESSORS, 
                time_horizon = TIME_HORIZON, 
                criticality_weights_set = CRITICAILITY_WEIGHTS_SET, 
                evaluation_weights = EVALUATION_WEIGHTS, 
                number_of_scenarios = NUM_SCENARIOS, 
                discounting_factor = DISCOUNTING_FACTOR,
                overflow_criteria = OVERFLOW_CRITERIA,
                starvation_criteria = STARVATION_CRITERIA,
                swap_threshold = BATTERY_LIMIT_TO_SWAP
                 ):
        self.max_depth = max_depth
        self.number_of_successors = number_of_successors
        self.time_horizon = time_horizon
        self.criticality_weights_set = criticality_weights_set
        self.evaluation_weights = evaluation_weights
        self.number_of_scenarios = number_of_scenarios
        self.discounting_factor = discounting_factor
        self.overflow_criteria = overflow_criteria
        self.starvation_criteria = starvation_criteria
        self.swap_threshold = swap_threshold
        super().__init__()
    
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
        escooters_to_pickup = []
        escooters_to_deliver = []
        batteries_to_swap = []

        end_time = simul.time + self.time_horizon 
        total_num_bikes_in_system = len(simul.state.get_all_ff_bikes())

        # Goes to depot if the vehicle's battery inventory is empty on arrival, and picks up all escooters at location that is unusable
        if vehicle.battery_inventory <= 0 and len(simul.state.depots) > 0:
            next_location = simul.state.get_closest_depot(simul, vehicle)
            escooters_to_pickup = [escooter.bike_id for escooter in vehicle.cluster.get_bikes() if escooter.usable()]
            max_pickup = min(vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()), len(escooters_to_pickup))
            return sim.Action(
                [],
                escooters_to_pickup[:max_pickup],
                [],
                next_location
            )

        start_time = time.time()
        # Loading and swap strategy at current location or cluster is always chosen greedily
        if vehicle.cluster is None:
            escooters_to_pickup, escooters_to_deliver, batteries_to_swap = calculate_loading_quantities_and_swaps_greedy(vehicle, simul, vehicle.location, self.overflow_criteria, self.starvation_criteria, self.swap_threshold)
            number_of_escooters_pickup = len(escooters_to_pickup)
            number_of_escooters_deliver = len(escooters_to_deliver)
            number_of_batteries_to_swap = len(batteries_to_swap)
        else:
            escooters_to_pickup, escooters_to_deliver, batteries_to_swap = calculate_loading_quantities_and_swaps_greedy(vehicle, simul, vehicle.cluster, self.overflow_criteria, self.starvation_criteria, self.swap_threshold)
            number_of_escooters_pickup = len(escooters_to_pickup)
            number_of_escooters_deliver = len(escooters_to_deliver)
            number_of_batteries_to_swap = len(batteries_to_swap)
        print("time to find action FF:", time.time() - start_time)

        # TODO v.location eller v.cluster??
        # Make a plan for all vehicles
        start_time = time.time()
        plan_dict = dict()
        for v in simul.state.get_ff_vehicles():
            # If vehicle is at a location, add current location to the plan with the greedy loading and swap strategy
            if v.eta == 0:
                plan_dict[v.vehicle_id] = [Visit(v.location, number_of_escooters_pickup, number_of_escooters_deliver, number_of_batteries_to_swap, simul.time, v)]

            # If the vehicle is driving, use pilot to calculate the loading and swap strategy and add to the plan
            else:
                number_of_escooters_pickup, number_of_escooters_deliver, number_of_batteries_to_swap = self.calculate_loading_quantities_and_swaps_pilot(v, simul, v.location, v.eta) #I think eta is estimated time of arrival
                plan_dict[v.vehicle_id] = [Visit(v.location, int(number_of_escooters_pickup), int(number_of_escooters_deliver), int(number_of_batteries_to_swap), v.eta, v)]
        
        # All locations the vehicles are at or are on their way to is added to the tabu list and plan
        tabu_list = [v.location.location_id for v in simul.state.get_ff_vehicles()]
        plan = Plan(plan_dict, tabu_list)

        # Use X-PILOT-BS to find which location to drive to next
        next_location, cluster = self.PILOT_function(simul, vehicle, plan, self.max_depth, self.number_of_successors, end_time, total_num_bikes_in_system)
        print("time to find next loc FF:", time.time() - start_time)

        # Count the neighbors that are starved or congested
        similary_imbalances_starved = 0
        similary_imbalances_overflow = 0
        # TODO Trenger endringer
        # for neighbor in simul.state.clusters[next_location].neighboring_clusters:
        #     if neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(BATTERY_LIMIT_TO_USE)) > self.overflow_criteria * neighbor.get_target_state(simul.day(),simul.hour()) and number_of_escooters_pickup > 0:
        #         similary_imbalances_overflow += 1
        #     elif neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(BATTERY_LIMIT_TO_USE)) < self.starvation_criteria * neighbor.get_target_state(simul.day(),simul.hour()) and number_of_escooters_deliver > 0:
        #         similary_imbalances_starved += 1
        # simul.metrics.add_aggregate_metric(simul, "similarly imbalanced starved", similary_imbalances_starved)
        # simul.metrics.add_aggregate_metric(simul, "similarly imbalanced congested", similary_imbalances_overflow)
        # simul.metrics.add_aggregate_metric(simul, "accumulated solution time", time.time()-start_logging_time)
        # simul.metrics.add_aggregate_metric(simul, 'number of problems solved', 1)

        return sim.Action(
            batteries_to_swap,
            escooters_to_pickup,
            escooters_to_deliver,
            next_location,
            cluster
        )

    def PILOT_function(self, simul, vehicle, initial_plan, max_depth, number_of_successors, end_time, total_num_bikes_in_system):
        """
        Returns an id of the next location the vehicle should drive to next, based on possible future scenarios and the outcome that happens if this location is visited.

        Parameters:
        - simul = Simulator
        - vehicle = Vehicle-object the decision is made for
        - initial_plan = Plan-object with the first visit for each vehicle and the tabu list (Each vehicle has a list with length = 1)
        - max_depth = Number of times to consider multiple actions and outcomes (after this every further move is chosen greedily)
        - number_of_successors = Number of possible moves that is considered for each location
        - end_time = Time horizon to evaluate
        - total_num_bikes_in_system = Total number of bikes in system
        """
        
        # Create a tree of possible plans, each are evaluated with different criticality weights
        completed_plans = []
        for weight_set in self.criticality_weights_set:
            plans = [[] for i in range(max_depth +1)]
            plans[0].append(initial_plan)

            for depth in range(1, max_depth+1):
                # Halve the branching width for each depth 
                if depth > 1:
                    number_of_successors = max(1, number_of_successors//2)
                
                # Explore as long as there are plans at the current depth
                while plans[depth-1] != []:
                    plan = plans[depth-1].pop(0)
                    next_vehicle = plan.next_visit.vehicle

                    # If the next vehicle is not the vehicle considered, reduce the number of successors
                    if next_vehicle != vehicle:
                        num_successors_other_veheicle = max(1, round(number_of_successors/2))
                        new_visits = self.greedy_next_visit(plan, simul, num_successors_other_veheicle, weight_set, total_num_bikes_in_system)
                    else:
                        new_visits = self.greedy_next_visit(plan, simul, number_of_successors, weight_set, total_num_bikes_in_system)
                    
                    # If there are no new visits or there is no visit within the time frame, finalize the plan
                    if new_visits == None or plan.next_visit.get_depature_time() > end_time:
                        new_plan = Plan(plan.copy_plan(),copy_arr_iter(plan.tabu_list), weight_set, plan.branch_number)
                        plans[depth].append(new_plan)
                    else:
                        # Otherwise, for each new visit, create a new branch of the plan
                        for branch_number, visit in enumerate(new_visits):
                            new_plan_dict = plan.copy_plan()
                            new_plan_dict[next_vehicle.vehicle_id].append(visit)
                            tabu_list = copy_arr_iter(plan.tabu_list)
                            tabu_list.append(visit.station.location_id)

                            # Create a new plan with updated information
                            if depth == 1:
                                new_plan = Plan(new_plan_dict, tabu_list, weight_set, branch_number)
                            else:
                                new_plan = Plan(new_plan_dict, tabu_list, weight_set, plan.branch_number)
                            
                            # Add the new plan to the appropriate list for further exploration
                            if next_vehicle.vehicle_id == vehicle.vehicle_id:
                                plans[depth].append(new_plan)
                            else:
                                plans[depth-1].append(new_plan)

            # Extend the plans until the end time is reached 
            for plan in plans[max_depth]:

                dep_time = plan.next_visit.get_depature_time()
                temp_plan = Plan(plan.copy_plan(), copy_arr_iter(plan.tabu_list), weight_set, plan.branch_number)

                if dep_time > end_time:
                    print(f'dep_time({dep_time}) > end_time({end_time}), when time_horizon={self.time_horizon}')
                    dep_time = end_time - 1

                # Add more visits until departure time has reached the end time
                while dep_time < end_time:
                    new_visit = self.greedy_next_visit(temp_plan, simul, 1, weight_set, total_num_bikes_in_system)
    
                    if new_visit != None:
                        new_visit = new_visit[0]
                        temp_plan.tabu_list.append(new_visit.station.location_id)
                        temp_plan.plan[temp_plan.next_visit.vehicle.vehicle_id].append(new_visit)
                        dep_time = new_visit.get_depature_time()
                        temp_plan.find_next_visit()
                    else:
                        break
                
                
                completed_plans.append(temp_plan)
        
        # Give a score for each plan, based on different demand scenarios
        plan_scores = dict()
        scenarios = self.generate_scenarioes(simul, self.number_of_scenarios, poisson = True)
        for plan in completed_plans:
            plan_scores[plan] = []
            for scenario_dict in scenarios:
                score = 0
                for v in plan.plan:
                    score += self.evaluate_route(plan.plan[v], scenario_dict, end_time, simul, self.evaluation_weights, total_num_bikes_in_system)
                plan_scores[plan].append(score)
        
        # Returns the center and cluster with the best average score over all scenarios
        return self.return_best_move_average(vehicle, simul, plan_scores)

    def calculate_loading_quantities_and_swaps_pilot(self, vehicle, simul, cluster, eta):
        """
        Returns the NUMBER of bikes to pick up, deliver and swap batteries on. Takes future demand into account by calculating it for the next hour, and treats it as evenly distributed throughout that hour

        Parameters:
        - vehicle = Vehicle-object the action is considered for
        - simul = Simulator
        - cluster = cluster the vehicle is considering doing the action at
        - eta = Estimated time of arrival for the vehicle to arrive at cluster
        """
        num_escooters_vehicle = len(vehicle.get_ff_bike_inventory())
        number_of_escooters_pickup = 0
        number_of_escooters_deliver = 0
        number_of_escooters_swap = 0

        target_state = round(cluster.get_target_state(simul.day(),simul.hour())) 
        time_until_arrival = eta - simul.time
        net_demand = calculate_net_demand(cluster, simul.time, simul.day(), simul.hour(), min(time_until_arrival, 60))
        max_num_usable_escooters = get_max_num_usable_escooters(cluster, vehicle)
        max_num_usable_escooters_eta = max_num_usable_escooters + net_demand

        # TODO check this and down
        neighbour_correction = 0
        for neighbor in cluster.neighbours:
            net_demand_neighbor = calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(), min(60, time_until_arrival))
            num_usable_bikes_neighbor_eta = len(neighbor.get_available_bikes()) + net_demand_neighbor
            neighbor_target_state = round(neighbor.get_target_state(simul.day(), simul.hour()))
            neighbour_correction += neighbor_target_state - num_usable_bikes_neighbor_eta

        # Calculate how many escooters to do different actions on
        if max_num_usable_escooters_eta < target_state: # delivery
            number_of_additional_escooters = min(num_escooters_vehicle, target_state - max_num_usable_escooters_eta + round(neighbour_correction))
            escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = get_escooter_ids_load_swap(cluster, vehicle, number_of_additional_escooters, "deliver", self.swap_threshold)
            number_of_escooters_deliver = len(escooters_to_deliver_accounted_for_battery_swaps)
            number_of_escooters_swap = len(escooters_to_swap_accounted_for_battery_swap)
        elif max_num_usable_escooters_eta > target_state: # pickups
            remaining_cap_vehicle = vehicle.bike_inventory_capacity - num_escooters_vehicle
            number_of_less_escooters = min(remaining_cap_vehicle, max_num_usable_escooters_eta - target_state - round(neighbour_correction))
            escooters_to_pickup_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = get_escooter_ids_load_swap(cluster, vehicle, number_of_less_escooters, "pickup", self.swap_threshold)
            number_of_escooters_pickup = len(escooters_to_pickup_accounted_for_battery_swaps)
            number_of_escooters_swap = len(escooters_to_swap_accounted_for_battery_swap)
        else: # only swap
            escooters_in_cluster_low_battery = cluster.get_swappable_bikes(BATTERY_LIMIT_TO_USE)
            number_of_escooters_swap = min(len(escooters_in_cluster_low_battery), vehicle.battery_inventory)
        
        return number_of_escooters_pickup, number_of_escooters_deliver, number_of_escooters_swap
    
    def greedy_next_visit(self, plan, simul, number_of_successors, weight_set, total_num_bikes_in_system):
        """
        Returns a list of Visits, greedily generated based on criticality scores.

        Parameters:
        - plan = Plan made so far from the PILOT method
        - simul = Simulator
        - number_of_successors = number of locations to consider
        - weight_set = criticality weights
        - total_num_bikes_in_system = total number of bikes in the system
        """
        visits = []
        tabu_list = plan.tabu_list
        vehicle = plan.next_visit.vehicle
        num_bikes_now = len(vehicle.get_ff_bike_inventory())

        # Update the vehicle bike inventory based on the planned operational actions
        for visit in plan.plan[vehicle.vehicle_id]:
            num_bikes_now += visit.loading_quantity
            num_bikes_now -= visit.unloading_quantity

        # Finds potential next clusters based on pick up or delivery status of the cluster and tabulist
        potential_clusters = find_potential_clusters(simul, 0.15, vehicle, num_bikes_now) # TODO margin in settings?
        if potential_clusters == []:
            return None
        
        number_of_successors = min(number_of_successors, len(potential_clusters))

        # Finds the criticality score of all potential clusters, and sort them in descending order
        clusters_sorted = calculate_criticality(weight_set, simul, potential_clusters, plan.plan[vehicle.vehicle_id][-1].station, total_num_bikes_in_system ,tabu_list)
        clusters_sorted_list = list(clusters_sorted.keys())
        
        # Selects the most critical clusters as next visits
        next_clusters = [clusters_sorted_list[i] for i in range(number_of_successors)]

        for next_cluster in next_clusters:
            arrival_time = plan.plan[vehicle.vehicle_id][-1].get_depature_time() + simul.state.get_travel_time(plan.plan[vehicle.vehicle_id][-1].station.location_id, next_cluster.location_id) + MINUTES_CONSTANT_PER_ACTION
            number_of_escooters_to_pickup, number_of_escooters_to_deliver, number_of_escooters_to_swap = self.calculate_loading_quantities_and_swaps_pilot(vehicle, simul, next_cluster, arrival_time)
            new_visit = Visit(next_cluster, number_of_escooters_to_pickup, number_of_escooters_to_deliver, number_of_escooters_to_swap, arrival_time, vehicle)
            visits.append(new_visit)
        
        return visits

    def generate_scenarioes(self, simul, number_of_scenarios, poisson = True): #normal_dist if poisson = False
        """
        Returns a list of generated scenarios.
        
        Parameters:
        - simul = Simulator
        - number_of_scenarios = numbers of scenarios to generate
        - poisson = Uses poisson distribution if True, normal distribution if False
        """
        rng = np.random.default_rng(simul.state.seed) 
        scenarios = []
        locations_dict = simul.state.locations
        if number_of_scenarios < 1:
            scenario_dict = dict()
            for area_id in locations_dict:
                net_demand =  calculate_net_demand(locations_dict[area_id], simul.time ,simul.day(),simul.hour(), 60)
                scenario_dict[area_id] = net_demand
            scenarios.append(scenario_dict)
        
        else:
            for s in range(number_of_scenarios):
                scenario_dict = dict()
                planning_horizon = self.time_horizon
                time_now = simul.time
                day = simul.day()
                hour = simul.hour()
                minute_in_current_hour = time_now - day*24*60 - hour*60
                minutes_current_hour = min(60-minute_in_current_hour,planning_horizon)
                minutes_next_hour = planning_horizon - minutes_current_hour
                
                next_hour = (hour + 1) % 24
                next_day = day if next_hour > hour else (day + 1) % 7

                # Make a dictionary deciding the expected net_nemand for each area
                # key = area_id, value = net_demand (either decided through poisson or normal distribution)
                for area_id in locations_dict: 
                    expected_arrive_intensity = locations_dict[area_id].get_arrive_intensity(day, hour)
                    expected_leave_intensity = locations_dict[area_id].get_leave_intensity(day, hour)
                    expected_arrive_intensity_next = locations_dict[area_id].get_arrive_intensity(next_day, next_hour)
                    expected_leave_intensity_next = locations_dict[area_id].get_leave_intensity(next_day, next_hour)
                    
                    if poisson:
                        net_demand_current = rng.poisson(expected_arrive_intensity) - rng.poisson(expected_leave_intensity)
                        net_demand_next = rng.poisson(expected_arrive_intensity_next) - rng.poisson(expected_leave_intensity_next)
                        net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/planning_horizon
                    
                    else: #normal_dist
                        arrive_intensity_stdev = locations_dict[area_id].get_arrive_intensity_stdev(day, hour)
                        leave_intensity_stdev = locations_dict[area_id].get_leave_intensity_stdev(day, hour)
                        arrive_intensity_stdev_next = locations_dict[area_id].get_arrive_intensity_stdev(next_day, next_hour)
                        leave_intensity_stdev_next = locations_dict[area_id].get_leave_intensity_stdev(next_day, next_hour)

                        net_demand_current = rng.normal(expected_arrive_intensity, arrive_intensity_stdev) - rng.normal(expected_leave_intensity, leave_intensity_stdev)
                        net_demand_next = rng.normal(expected_arrive_intensity_next, arrive_intensity_stdev_next) - rng.normal(expected_leave_intensity_next, leave_intensity_stdev_next)
                        net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/planning_horizon
                        
                    scenario_dict[area_id] = net_demand 
                scenarios.append(scenario_dict)
        
        # Return a list of num_scenarios dictionaries with expected net demand for each area in the future
        return scenarios
    
    def evaluate_route(self, route, scenario_dict, end_time, simul, weights, total_num_bikes_in_system):
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
        # TODO sanity check
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

            area = visit.station
            neighbors = area.neighbours

            eta = visit.arrival_time

            if eta > end_time:
                eta = end_time
            
            initial_inventory = len(area.get_available_bikes())
            net_demand = scenario_dict[area.location_id]
            target_state = area.get_target_state(simul.day(), simul.hour())

            # Calculate when the first starvation or congestion will occur if not visited
            if net_demand < 0:
                sorted_escooters_at_area = sorted(area.get_bikes(), key=lambda bike: bike.battery, reverse=False)
                time_first_violation_no_visit = current_time + min((area.number_of_bikes() - len(area.get_swappable_bikes(BATTERY_LEVEL_LOWER_BOUND)))/ -net_demand, (sum(Ebike.battery for Ebike in sorted_escooters_at_area[-3:])/3)/(calculate_hourly_discharge_rate(simul, total_num_bikes_in_system)*60))
            else:
                time_first_violation_no_visit = end_time
            
            # Number of violations that happen if we dont visit the area
            # Violation_no_visit negative if starvation 
            if end_time > time_first_violation_no_visit:
                violation_no_visit = ((end_time - time_first_violation_no_visit)/60) * net_demand
            else:
                violation_no_visit = 0
            
            # Violations that we cant avoide due to driveing time
            if eta > time_first_violation_no_visit:
                unavoidable_violations = ((eta - time_first_violation_no_visit)/60) * net_demand
            else:
                unavoidable_violations = 0
            
            # Number of escooters at area after visit
            inventory_after_loading_and_swaps = initial_inventory + ((eta - current_time)/60)*net_demand - unavoidable_violations - loading_quantity + unloading_quantity + swap_quantity
            
            # Time for first violation if we visit
            if net_demand < 0:
                if swap_quantity > loading_quantity+2:
                    time_first_violation_after_visit = eta + min((inventory_after_loading_and_swaps/(-net_demand))*60, 100/calculate_hourly_discharge_rate(simul, total_num_bikes_in_system))
                else:
                    time_first_violation_after_visit = eta + min((inventory_after_loading_and_swaps/(-net_demand))*60, (sum(ebike.battery for ebike in sorted_escooters_at_area[-3:])/3)/(calculate_hourly_discharge_rate(simul, total_num_bikes_in_system)*60))
            else:
                time_first_violation_after_visit = end_time
            
            if time_first_violation_after_visit < end_time:
                violations_after_visit = ((end_time - time_first_violation_after_visit)/60) * net_demand
            else:
                violations_after_visit = 0

            
            avoided_violations = violation_no_visit - violations_after_visit

            ending_inventory = max(0, inventory_after_loading_and_swaps + ((end_time - eta)/60) * net_demand)
            deviation_visit = abs(ending_inventory - target_state)
   
            ending_inventory_no_visit = max(0, initial_inventory + ((end_time - current_time)/60) * net_demand)
            deviation_no_visit = abs(ending_inventory_no_visit - target_state)

            improved_deviation = deviation_no_visit - deviation_visit

            excess_escooters = ending_inventory
            excess_escooters_no_visit = ending_inventory_no_visit

            expected_number_of_escooters = inventory_after_loading_and_swaps
            area_type = calculate_cluster_type(target_state, expected_number_of_escooters)

            for neighbor in neighbors:
                roamings = 0
                roamings_no_visit = 0
                net_demand_neighbor = scenario_dict[neighbor.location_id]
                expected_ecooters_neighbor = neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(BATTERY_LEVEL_LOWER_BOUND)) + net_demand_neighbor
                neighbor_type = calculate_cluster_type(neighbor.get_target_state(simul.day(),simul.hour()),expected_ecooters_neighbor)

                if neighbor_type == area_type:
                    if net_demand_neighbor < 0:
                        time_first_violation = current_time + ((neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(BATTERY_LEVEL_LOWER_BOUND)))/-net_demand_neighbor) * 60
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
                        
            
                distance_scaling = ((simul.state.get_vehicle_travel_time(area.location_id, neighbor.location_id)/60)* VEHICLE_SPEED)/MAX_ROAMING_DISTANCE_SOLUTIONS
                neighbor_roamings += (1-distance_scaling)*roamings-roamings_no_visit
            
            avoided_disutility += discounting_factors[counter]*(weights[0]*avoided_violations + weights[1]*neighbor_roamings + weights[2]*improved_deviation)

            counter += 1
        
        return avoided_disutility
    
    def return_best_move_average(self, vehicle, simul, plan_scores):
        """
        Returns the ID of the Area with performing best on average over all the scenarios.

        Parameters:
        - vehicle = The Vehicle-object doing the action
        - simul = Simulator
        - plan_scores = dictionaries, key: Plan, value: list of float-scores for each scenario
        """
        
        if self.number_of_scenarios==0:
            self.number_of_scenarios = 1
        
        # Make a dictionary with a value of the sum instead of a list of scores
        score_board = dict()
        for plan in plan_scores:
            score_board[plan] = sum(plan_scores[plan][scenario_id] for scenario_id in range(self.number_of_scenarios))

        # Sort the score board in descending order by scores
        score_board_sorted = dict(sorted(score_board.items(), key=lambda item: item[1], reverse=True))
        
        # If there is a best plan return that one
        if list(score_board_sorted.keys())[0] != None:
            best_plan = list(score_board_sorted.keys())[0]
            branch = best_plan.branch_number
            if branch is None: # TODO dette skjer når den ikke klarer å lage plan med neste stopp
                print('branch is None')
                first_move = simul.state.get_closest_depot(vehicle)
                cluster = None
                return first_move, cluster
            first_move = best_plan.plan[vehicle.vehicle_id][1].station.location_id
            cluster = best_plan.plan[vehicle.vehicle_id][1].station
            return first_move, cluster
        
        # If there is no best, choose a random station that is not in the tabu_list
        else: 
            tabu_list = [vehicle2.location.location_id for vehicle2 in simul.state.get_ff_vehicles()]
            potential_stations2 = [station for station in simul.state.get_areas() if station.location_id not in tabu_list]    
            rng_balanced = np.random.default_rng(None)
            cluster = rng_balanced.choice(potential_stations2)
            return cluster.location_id, cluster

def calculate_loading_quantities_and_swaps_greedy(vehicle, simul, station, overflow_criteria, starvation_criteria, swap_threshold):
    """
    Returns a list of IDs of the bikes to deliver, pickup or swap batteries on.
    The calculation is done when a vehicle arrives at the station, and the list returned are performed.

    Parameters:
    - vehicle = Vehicle-object that is doing the action
    - simul = Simulator
    - station = The Station-object under consideration
    - congestion_criteria = percentage of station capacity for a station to be considered congested
    - starvation_critera = percentage of station capacity for a station to be considered starved
    """
    # TODO sanity check
    target_state = round(station.get_target_state(simul.day(), simul.hour())) #Denne må vi finne ut hvordan lages
    num_escooters_accounted_for_battery_swaps = get_max_num_usable_escooters(station, vehicle) # 

    starved_neighbors = 0
    overflowing_neighbors = 0

    for neighbor in station.neighbours:
        num_escooters_neighbor = neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(BATTERY_LEVEL_LOWER_BOUND))
        neighbor_target_state = round(neighbor.get_target_state(simul.day(), simul.hour()))
        if num_escooters_neighbor < starvation_criteria * neighbor_target_state:
            starved_neighbors += 1
        elif num_escooters_neighbor > overflow_criteria * neighbor_target_state:
            overflowing_neighbors += 1

    if num_escooters_accounted_for_battery_swaps < target_state: #Ta hensyn til nabocluster her?
        number_of_escooters_to_deliver = min(len([escooter for escooter in vehicle.get_ff_bike_inventory() if escooter.battery > BATTERY_LEVEL_LOWER_BOUND]), target_state - num_escooters_accounted_for_battery_swaps + BIKES_STARVED_NEIGHBOR * starved_neighbors) # discuss 2*starved_neighbors part, ta hensyn til postensielle utladede scootere i bilen
        escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = get_escooter_ids_load_swap(station, vehicle, number_of_escooters_to_deliver, "deliver", swap_threshold)
        escooters_to_pickup_accounted_for_battery_swaps = []
    
    elif num_escooters_accounted_for_battery_swaps > target_state:
        remaining_cap_vehicle = vehicle.bike_inventory_capacity - len(vehicle.get_ff_bike_inventory())
        number_of_escooters_to_pickup = min(remaining_cap_vehicle, num_escooters_accounted_for_battery_swaps - target_state + BIKES_OVERFLOW_NEIGHBOR * overflowing_neighbors, len(station.get_bikes())) #discuss logic behind this
        escooters_to_deliver_accounted_for_battery_swaps=[]
        escooters_to_pickup_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = get_escooter_ids_load_swap(station, vehicle, number_of_escooters_to_pickup, "pickup", swap_threshold)

    else:
        escooters_to_pickup_accounted_for_battery_swaps = []
        escooters_to_deliver_accounted_for_battery_swaps = []

        escooters_in_station_low_battery = station.get_swappable_bikes(BATTERY_LIMIT_TO_USE)
        num_escooters_to_swap = min(len(escooters_in_station_low_battery),vehicle.battery_inventory)
        escooters_to_swap_accounted_for_battery_swap = [escooter.bike_id for escooter in escooters_in_station_low_battery[:num_escooters_to_swap]]
    
    return escooters_to_pickup_accounted_for_battery_swaps, escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap


def get_max_num_usable_escooters(cluster, vehicle): 
    """"
    Returns max number of bikes at station with sufficient battery level, neglects bikes that cannot be fixed.

    Parameters:
    - cluster = Cluster-object being considered
    - vehicle = Vehicle considered to rebalance station
    """
    return len(cluster.get_available_bikes()) + min(len(cluster.get_swappable_bikes()), vehicle.battery_inventory)

def get_escooter_ids_load_swap(cluster, vehicle, number_of_escooters, cluster_type, swap_threshold):
    """
    Returns lists of the IDs of the bikes to deliver/pick-up and swap.

    Parameters:
    - station = Station being considered
    - vehicle = Vehicle doing the action
    - num_bikes = difference from target state after battery swap on site is done + effects of neighbors
    - station_type = if there has to be unloading or pick-ups done at the station
    """
    if SORTED_BIKES:
        escooters_at_cluster = sorted(cluster.get_bikes(), key=lambda bike: bike.battery, reverse=False)
        escooters_in_vehicle =  sorted(vehicle.get_ff_bike_inventory(), key=lambda bike: bike.battery, reverse=False)
    else:
        escooters_at_cluster = cluster.get_bikes()
        escooters_in_vehicle =  vehicle.get_ff_bike_inventory()


    if cluster_type == "deliver":
        number_of_escooters_to_swap = min(len(cluster.get_swappable_bikes(swap_threshold)),vehicle.battery_inventory)
        #TODO Burde vi runde opp eller ned?
        number_of_escooters_to_deliver = int(number_of_escooters)+1

        escooters_to_swap = [escooter.bike_id for escooter in escooters_at_cluster[:number_of_escooters_to_swap]]
        escooters_to_deliver = [escooter.bike_id for escooter in escooters_in_vehicle[-number_of_escooters_to_deliver:]]

        return escooters_to_deliver, escooters_to_swap
    
    elif cluster_type == "pickup":
        number_of_escooters_to_swap_and_pickup = min(len(cluster.get_swappable_bikes(swap_threshold)), vehicle.battery_inventory, round(number_of_escooters))
        number_of_escooters_to_only_pickup = round(number_of_escooters) - number_of_escooters_to_swap_and_pickup # how many to pick up
        number_of_escooters_to_only_swap = min(vehicle.battery_inventory, max(0,len(cluster.get_swappable_bikes(BATTERY_LIMIT_TO_USE)) - number_of_escooters_to_swap_and_pickup))

        escooters_to_swap = []
        escooters_to_pickup = [escooter.bike_id for escooter in escooters_at_cluster[:number_of_escooters_to_swap_and_pickup]]
        if number_of_escooters_to_only_pickup > 0:
            escooters_to_pickup += [escooter.bike_id for escooter in escooters_at_cluster[-number_of_escooters_to_only_pickup:]]
        
        elif number_of_escooters_to_only_swap > 0:
            escooters_to_swap += [escooter.bike_id for escooter in escooters_at_cluster[number_of_escooters_to_swap_and_pickup:number_of_escooters_to_swap_and_pickup+number_of_escooters_to_only_swap]]

        return escooters_to_pickup, escooters_to_swap
    
    return [],[]

def find_potential_clusters(simul, cutoff_vehicle, vehicle, bikes_at_vehicle):
    """
    Returns a list of Station-Objects that are not in the tabu list, and that need help to reach target state.

    Parameters:
    - simul = Simulator
    - cutoff_vehicle = At what percentage of the vehicle's capacity is the vehicle considered "empty" or "full"
    - vehicle = Vehicle-object under consideration
    - bikes_at_vehicle = number of bikes in the vehicle at the time
    """
    potential_pickup_stations = []
    potential_delivery_stations = []

    if cutoff_vehicle * vehicle.bike_inventory_capacity <= bikes_at_vehicle <= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
        potential_pickup_stations = clusterPickup(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, MAX_WALKING_AREAS, vehicle, simul)
        potential_delivery_stations = clusterDelivery(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, MAX_WALKING_AREAS, vehicle, simul)

        potential_stations = potential_pickup_stations + potential_delivery_stations
    else:
        if bikes_at_vehicle <= cutoff_vehicle*vehicle.bike_inventory_capacity:
            potential_stations = clusterPickup(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, MAX_WALKING_AREAS, vehicle, simul)

        elif bikes_at_vehicle >= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
            potential_stations = clusterDelivery(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, MAX_WALKING_AREAS, vehicle, simul)

    return potential_stations