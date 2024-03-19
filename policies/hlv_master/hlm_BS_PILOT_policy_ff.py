from policies import Policy
from settings import *
import sim
from .Visit import Visit
from .Plan import Plan
from .Criticality_score_ff import calculate_criticality, calculate_station_type
from .Simple_calculations import calculate_net_demand, copy_arr_iter, generate_discounting_factors, calculate_hourly_discharge_rate
from .dynamic_clustering import clusterPickup, clusterDelivery

import numpy as np
import time

class BS_PILOT_FF(Policy):
    """
    Class for X-PILOT-BS policy. Finds the best action for e-scooter systems.
    """
    def __init__(self, 
                max_depth = settings_max_depth, 
                number_of_successors = settings_number_of_successors, 
                time_horizon = TIME_HORIZON, 
                criticality_weights_set = SETTINGS_CRITICAILITY_WEIGHTS_SET, 
                evaluation_weights = settings_evaluation_weights, 
                number_of_scenarios = settings_number_of_scenarios, 
                discounting_factor = settings_discounting_factor,
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
    
    def get_best_action(self, simul: sim.Simulator, vehicle: sim.Vehicle):
        """
        Returns an Action (with which escooters to swap batteries on, which escooters to pick-up, which escooters to unload, next location to drive to)

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
        total_num_escooters_in_system = len(simul.state.get_all_ff_bikes())

        # Goes to depot if the vehicle's battery inventory is empty on arrival, and picks up all escooters at cluster that is unusable
        if vehicle.battery_inventory <= 0 and len(simul.state.get_ff_depots()) > 0:
            next_location = self.simul.state.get_closest_depot(vehicle)
            escooters_to_pickup = [escooter.bike_id for escooter in vehicle.cluster.bikes.values() if not escooter.usable()]
            max_pickup = min(vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()), len(escooters_to_pickup))
            return sim.Action(
                [],
                escooters_to_pickup[:max_pickup],
                [],
                next_location
            )

        # Loading and swap strategy at current location is always chosen greedily
        # If vehicle is not at a cluster, find escooters a the current location, otherwise find escooters in cluster
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


        # Make a plan for all vehicles
        plan_dict = dict()
        for v in simul.state.get_ff_vehicles():
            # If vehicle is at a location, add current location to the plan with the greedy loading and swap strategy
            if v.eta == 0: # TODO, skal vi bruke v.location og ikke v.cluster?
                plan_dict[v.vehicle_id] = [Visit(v.location, number_of_escooters_pickup, number_of_escooters_deliver, number_of_batteries_to_swap, simul.time, v)]
            
            # If the vehicle is driving, use pilot to calculate the loading and swap strategy and add to the plan
            else:
                number_of_escooters_pickup, number_of_escooters_deliver, number_of_batteries_to_swap = self.calculate_loading_quantities_and_swaps_pilot(v, simul, v.location, v.eta) 
                plan_dict[v.vehicle_id] = [Visit(v.location, int(number_of_escooters_pickup), int(number_of_escooters_deliver), int(number_of_batteries_to_swap), v.eta, v)]
        
        # All locations the vehicles are at or are on their way to is added to the tabu list and plan
        tabu_list = [v.location.location_id for v in simul.state.get_ff_vehicles()] # TODO skal alle areas i clustere være i listen?
        plan = Plan(plan_dict, tabu_list)

        # Use X-PILOT-BS to find which location to drive to next
        next_location, next_cluster = self.PILOT_function(simul, vehicle, plan, self.max_depth, self.number_of_successors, end_time, total_num_escooters_in_system)

        # Count the neighbors that are starved or congested
        similary_imbalances_starved = 0
        similary_imbalances_overflow = 0

        #TODO Trenger endringer
        # for neighbor in simul.state.stations[next_location].neighbours:
        #     if neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(BATTERY_LIMIT_TO_USE)) > self.overflow_criteria * neighbor.get_target_state(simul.day(),simul.hour()) and number_of_escooters_pickup > 0:
        #         similary_imbalances_overflow += 1
        #     elif neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(BATTERY_LIMIT_TO_USE)) < self.starvation_criteria * neighbor.get_target_state(simul.day(),simul.hour()) and number_of_escooters_deliver > 0:
        #         similary_imbalances_starved += 1
            
        simul.metrics.add_aggregate_metric(simul, "similarly imbalanced starved", similary_imbalances_starved)
        simul.metrics.add_aggregate_metric(simul, "similarly imbalanced congested", similary_imbalances_overflow)
        simul.metrics.add_aggregate_metric(simul, "accumulated solution time", time.time()-start_logging_time)
        simul.metrics.add_aggregate_metric(simul, 'number of problems solved', 1)

        if COLLAB_POLICY and vehicle.cluster is not None and next_cluster is not None:
            # Calculate how many bikes are not needed at current stations, and needed at next stations
            # TODO Hvis den ene lacker og den andre har for mange så balanseres de, og da trenger vi ikke å gjøre noe??
            current_stations = {area.station: area.station.number_of_bikes() - area.station.get_target_state(simul.day(), simul.hour()) for area in vehicle.cluster.areas if area.station is not None}
            overflow = sum(current_stations.values())

            next_stations = [area.station for area in next_cluster.areas if area.station is not None]
            lacking = sum([station.get_target_state(simul.day(), simul.hour()) - station.number_of_bikes() for station in next_stations])

            if overflow > 0 and lacking > 0:
                left_bike_capacity = vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()) - len(escooters_to_pickup)
                num_escooters_to_pickup = min(left_bike_capacity, overflow, lacking)
                bikes_to_pickup = []
                for station, deviation_from_target in current_stations.items():
                    if deviation_from_target > 0:
                        num_escooters = min(deviation_from_target, num_escooters_to_pickup)
                        pickup_ids, swap_ids = get_escooter_ids_load_swap(station, vehicle, num_escooters, "pickup")
                        bikes_to_pickup.extend(pickup_ids)
                        num_escooters_to_pickup -= num_escooters
                        # TODO denne må sjekkes haha

                return sim.Action(
                    batteries_to_swap,
                    escooters_to_pickup,
                    escooters_to_deliver,
                    next_location,
                    helping_pickup = bikes_to_pickup
                )


        return sim.Action(
            batteries_to_swap,
            escooters_to_pickup,
            escooters_to_deliver,
            next_location,
            next_cluster
        )

    def PILOT_function(self, simul, vehicle, initial_plan, max_depth, number_of_successors, end_time, total_num_escooters_in_system):
        """
        Returns an id of the next location the vehicle should drive to next, based on possible future scenarios and the outcome that happens if this location is visited.

        Parameters:
        - simul = Simulator
        - vehicle = Vehicle-object the decision is made for
        - initial_plan = Plan-object with the first visit for each vehicle and the tabu list (Each vehicle has a list with length = 1)
        - max_depth = Number of times to consider multiple actions and outcomes (after this every further move is chosen greedily)
        - number_of_successors = Number of possible moves that is considered for each location
        - end_time = Time horizon to evaluate
        - total_num_bikes_in_system = Total free-floating number of bikes
        """

        # Create a tree of possible plans, each are evaluated with different criticality weights
        completed_plans = []
        for weight_set in self.criticality_weights_set:
            plans = [[] for _ in range(max_depth+1)]
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
                        new_visits = self.greedy_next_visit(plan, simul, num_successors_other_veheicle, weight_set, total_num_escooters_in_system)
                    else:
                        new_visits = self.greedy_next_visit(plan, simul, number_of_successors, weight_set, total_num_escooters_in_system)
                    
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
                            tabu_list.append(visit.station.location_id) # TODO cluster eller station?

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
                
                # Add more visits until departure time has reached the end time
                while dep_time < end_time:
                    new_visit = self.greedy_next_visit(temp_plan, simul, 1, weight_set, total_num_escooters_in_system)
                    
                    if new_visit != None:
                        new_visit = new_visit[0]
                        temp_plan.tabu_list.append(new_visit.station.location_id) # TODO station?
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
                    score += self.evaluate_route(plan.plan[v], scenario_dict, end_time, simul, self.evaluation_weights, total_num_escooters_in_system)
                plan_scores[plan].append(score)
        
        # Returns the location with the best average score over all scenarios
        return self.return_best_move_average(vehicle, simul, plan_scores)

    def calculate_loading_quantities_and_swaps_pilot(self, vehicle, simul, cluster, eta):
        """
        Returns the NUMBER of bikes to pick up, deliver and swap batteries on. Takes future demand into account by calculating it for the next hour, and treats it as evenly distributed throughout that hour

        Parameters:
        - vehicle = Vehicle-object the action is considered for
        - simul = Simulator
        - cluster = Cluster the vehicle is considering doing the action at
        - eta = Estimated time of arrival for the vehicle to arrive at station
        """
        num_escooters_vehicle = len(vehicle.get_bike_inventory())
        num_pickup_escooters = 0
        num_deliver_escooters = 0
        num_swap_escooters = 0

        target_state = round(cluster.get_target_state(simul.day(),simul.hour())) 
        net_demand = calculate_net_demand(cluster, simul.time, simul.day(), simul.hour(), 60) #TODO Gives net_demand per hour or planning horizon?
        max_num_usable_escooters = get_max_num_usable_bikes(cluster, vehicle)
        # TODO self.time_horizon?
        max_num_usable_bikes_eta = max_num_usable_escooters + ((eta - simul.time)/60)*net_demand # How many bikes at station at eta, based on demand forecast

        # Calculate the amount of neighbors that are starving or congested, can impact the number of bikes to operate on
        num_starved_neighbors = 0
        num_overflowing_neighbors = 0
        for neighbor in cluster.get_neighbours():
            net_demand_neighbor = calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(),60) # TODO self.time_horizon?
            num_escooters_neighbor_eta = len(neighbor.get_available_bikes()) + ((eta - simul.time)/60)*net_demand_neighbor
            target_state_neighbor = round(neighbor.get_target_state(simul.day(), simul.hour()))
            if num_escooters_neighbor_eta < self.starvation_criteria * target_state_neighbor:
                num_starved_neighbors += 1
            elif num_escooters_neighbor_eta > self.overflow_criteria * target_state_neighbor:
                num_overflowing_neighbors += 1

        # Calculate how many bikes to do different actions on
        if max_num_usable_bikes_eta < target_state: # deliver
            number_of_additional_escooters = min(num_escooters_vehicle, target_state - max_num_usable_bikes_eta + BIKES_STARVED_NEIGHBOR * num_starved_neighbors)
            escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = get_escooter_ids_load_swap(cluster, vehicle, number_of_additional_escooters, "deliver")
            num_deliver_escooters = len(escooters_to_deliver_accounted_for_battery_swaps)
            num_swap_escooters = len(escooters_to_swap_accounted_for_battery_swap)
        
        elif max_num_usable_bikes_eta > target_state: # pick-up
            remaining_cap_vehicle = vehicle.bike_inventory_capacity - num_escooters_vehicle
            num_less_escooters = min(remaining_cap_vehicle, max_num_usable_bikes_eta - target_state + BIKES_OVERFLOW_NEIGHBOR * num_overflowing_neighbors)
            num_escooters_to_swap_and_pickup, escooters_to_swap_accounted_for_battery_swap = get_escooter_ids_load_swap(cluster, vehicle, num_less_escooters, "pickup")
            num_pickup_escooters = len(num_escooters_to_swap_and_pickup)
            num_swap_escooters = len(escooters_to_swap_accounted_for_battery_swap)

        else: # only swap
            escooters_in_station_low_battery = cluster.get_swappable_bikes(BATTERY_LIMIT_TO_USE)
            num_swap_escooters = min(len(escooters_in_station_low_battery), vehicle.battery_inventory)
        
        return num_pickup_escooters, num_deliver_escooters, num_swap_escooters
    
    def greedy_next_visit(self, plan, simul, number_of_successors, weight_set, total_num_escooters_in_system):
        """
        Returns a list of Visits, greedily generated based on criticality scores.

        Parameters:
        - plan = Plan made so far from the PILOT method
        - simul = Simulator
        - number_of_successors = number of locations to consider
        - weight_set = criticality weights
        - total_num_bikes_in_system = total number of escooters in the system
        """
        visits = []
        tabu_list = plan.tabu_list
        vehicle = plan.next_visit.vehicle

        num_bikes_now = len(vehicle.get_ff_bike_inventory()) #TODO bare ff?

        # Update the vehicle bike inventory based on the planned operational actions
        for visit in plan.plan[vehicle.vehicle_id]: #la til index -1 (gir dette mening?)
            num_bikes_now += visit.loading_quantity
            num_bikes_now -= visit.unloading_quantity

        # Finds potential next stations based on pick up or delivery status of the station and tabulist
        # TODO skal marginene være høyere? Trenger vi cluster_type?
        potential_clusters, cluster_type = find_potential_clusters(simul,VEHICLE_TYPE_MARGIN,vehicle, num_bikes_now)
        if potential_clusters == []: #Try to find out when and if this happens?
            return None
        
        number_of_successors = min(number_of_successors, len(potential_clusters))

        # Finds the criticality score of all potential stations, and sort them in descending order
        clusters_sorted = calculate_criticality(weight_set, simul, potential_clusters, plan.plan[vehicle.vehicle_id][-1].station,cluster_type, total_num_escooters_in_system ,tabu_list)
        clusters_sorted_list = list(clusters_sorted.keys())

        # Selects the most critical stations as next visits
        next_clusters = clusters_sorted_list[:number_of_successors]

        for next_cluster in next_clusters:
            arrival_time = plan.plan[vehicle.vehicle_id][-1].get_depature_time() + simul.state.traveltime_vehicle_matrix[(plan.plan[vehicle.vehicle_id][-1].station.location_id, next_cluster.location_id)] + MINUTES_CONSTANT_PER_ACTION
            number_of_escooters_to_pickup, number_of_escooters_to_deliver, number_of_escooters_to_swap = self.calculate_loading_quantities_and_swaps_pilot(vehicle, simul, next_cluster, arrival_time)
            new_visit = Visit(next_cluster, number_of_escooters_to_pickup, number_of_escooters_to_deliver, number_of_escooters_to_swap, arrival_time, vehicle)
            visits.append(new_visit)
        
        return visits
    
    def generate_scenarioes(self, simul, number_of_scenarios, poisson = True):
        """
        Returns a list of generated scenarios.
        
        Parameters:
        - simul = Simulator
        - number_of_scenarios = numbers of scenarios to generate
        - poisson = Uses poisson distribution if True, normal distribution if False
        """
        rng = np.random.default_rng(simul.state.seed) 
        scenarios = []
        locations_dict = simul.state.get_ff_locations()
        if number_of_scenarios < 1:
            scenario_dict = dict() 
            for location_id in locations_dict:
                net_demand =  calculate_net_demand(locations_dict[location_id], simul.time ,simul.day() ,simul.hour(), TIME_HORIZON)
                scenario_dict[location_id] = net_demand
            scenarios.append(scenario_dict)
        
        else:
            for s in range(number_of_scenarios):
                scenario_dict = dict()
                time_now = simul.time
                day = simul.day()
                hour = simul.hour()
                minute_in_current_hour = time_now - day*24*60 - hour*60
                minutes_current_hour = min(60 - minute_in_current_hour, self.time_horizon)
                minutes_next_hour = self.time_horizon - minutes_current_hour
                

                next_hour = (hour + 1) % 24
                next_day = day if next_hour != 0 else (day + 1) % 7

                # Make a dictionary deciding the expected net_nemand for each station
                # key = station_id, value = net_demand (either decided through poisson or normal distribution)
                for location_id in locations_dict: 
                    expected_arrive_intensity = locations_dict[location_id].get_arrive_intensity(day, hour)
                    expected_leave_intensity = locations_dict[location_id].get_leave_intensity(day, hour)
                    expected_arrive_intensity_next = locations_dict[location_id].get_arrive_intensity(next_day, next_hour)
                    expected_leave_intensity_next = locations_dict[location_id].get_leave_intensity(next_day, next_hour)

                    if poisson:
                        net_demand_current = rng.poisson(expected_arrive_intensity) - rng.poisson(expected_leave_intensity)
                        net_demand_next = rng.poisson(expected_arrive_intensity_next) - rng.poisson(expected_leave_intensity_next)
                        net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/self.time_horizon
                    
                    else: #normal_dist
                        arrive_intensity_stdev = locations_dict[location_id].get_arrive_intensity_stdev(day, hour)
                        leave_intensity_stdev = locations_dict[location_id].get_leave_intensity_stdev(day, hour)
                        arrive_intensity_stdev_next = locations_dict[location_id].get_arrive_intensity_stdev(next_day, next_hour)
                        leave_intensity_stdev_next = locations_dict[location_id].get_leave_intensity_stdev(next_day, next_hour)

                        net_demand_current = rng.normal(expected_arrive_intensity, arrive_intensity_stdev) - rng.normal(expected_leave_intensity, leave_intensity_stdev)
                        net_demand_next = rng.normal(expected_arrive_intensity_next, arrive_intensity_stdev_next) - rng.normal(expected_leave_intensity_next, leave_intensity_stdev_next)
                        net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/self.time_horizon

                    scenario_dict[location_id] = net_demand
                scenarios.append(scenario_dict)
        
        # Return a list of num_scenarios dictionaries with expected net demand for each station in the future
        return scenarios
    
    def evaluate_route(self, route, scenario_dict, end_time, simul, weights, total_num_escooters_in_system):
        """
        Returns the score based on if the vehicle drives this route in comparisson to not driving it at all.
        Evaluates based on avoided violations, neighbor roamings and improved deviation

        Parameters:
        - route = list of visits the vehicle is supposed to do
        - scenario_dict = a dictionary with a possible net demand for each station
        - end_time = the stopping point of the horizon to evaluate
        - simul = Simulator
        - weights = weights for avoided violations, neighbor roamings, and improved deviation
        - total_num_escooters_in_system = the total amount of escooters that are in the FF system
        """
        discounting_factors = generate_discounting_factors(len(route), self.discounting_factor)
        avoided_disutility = 0
        current_time = simul.time
        counter = 0 # which stage during the visit the vehicle is at -> used to discount the score

        # Summarized the score for each visit, and discounts it with a factor to "minimize" the impact on uncertain outcomes
        for visit in route:
            avoided_violations = 0
            neighbor_roamings = 0
            improved_deviation = 0

            area = visit.station

            loading_quantity = visit.loading_quantity
            unloading_quantity = visit.unloading_quantity
            swap_quantity = visit.swap_quantity
            eta = visit.arrival_time

            if eta > end_time:
                eta = end_time
            
            initial_inventory = len(area.get_available_bikes())
            net_demand = scenario_dict[area.location_id]
            target_state = area.get_target_state(simul.day(), simul.hour())

            # Calculate when the first starvation will occur if area is not visited
            if net_demand < 0:
                sorted_escooters_in_station = sorted(area.bikes.values(), key=lambda bike: bike.battery, reverse=False)

                # Calculate hours until violation because no bikes have sufficient battery
                battery_top3 = [escooter.battery for escooter in sorted_escooters_in_station[-3:]]
                average_battery_top3 = sum(battery_top3)/len(battery_top3) if battery_top3 else 0
                hourly_discharge = calculate_hourly_discharge_rate(simul, total_num_escooters_in_system)
                hours_until_violation_battery = average_battery_top3 / hourly_discharge

                # Find the earlist moment for a violation
                hours_until_first_violation = min(
                                                (len(area.get_available_bikes())/ -net_demand), # How long until the net demand results in a starvation
                                                hours_until_violation_battery
                                                )
                
                # Find the time in minutes for the violation
                time_first_violation_no_visit = current_time + (hours_until_first_violation * 60)
            else:
                time_first_violation_no_visit = end_time
            
            # Calculate number of violation within the time horizon
            if end_time > time_first_violation_no_visit:
                violation_no_visit = ((end_time - time_first_violation_no_visit)/60) * abs(net_demand)
            else:
                violation_no_visit = 0
            
            # Violations that we cant avoide due to driving time
            if eta > time_first_violation_no_visit:
                unavoidable_violations = ((eta - time_first_violation_no_visit)/60) * net_demand
            else:
                unavoidable_violations = 0
            
            # Number of escooters at station after visit
            area_inventory_after_visit = initial_inventory + ((eta - current_time)/60) * net_demand - unavoidable_violations - loading_quantity + unloading_quantity + swap_quantity
            
            # Time for first violation if we visit the station
            if net_demand < 0:
                time_until_first_violation = (area_inventory_after_visit / (-net_demand)) * 60
                if swap_quantity > loading_quantity + 3:
                    time_first_violation_after_visit = eta + min(time_until_first_violation, 100 / calculate_hourly_discharge_rate(simul, total_num_escooters_in_system, False))
                else:
                    time_first_violation_after_visit = eta + min(time_until_first_violation, (average_battery_top3)/(calculate_hourly_discharge_rate(simul, total_num_escooters_in_system)) * 60)
            else:
                time_first_violation_after_visit = end_time
            
            if time_first_violation_after_visit < end_time:
                violations_after_visit = ((end_time - time_first_violation_after_visit)/60) * abs(net_demand)
            else:
                violations_after_visit = 0

            # How many violations did we manage to avoid, not counting the ones we could not do anything about
            avoided_violations = violation_no_visit - violations_after_visit

            # Calculating the deviation from target at end time
            ending_inventory_after_visit = max(0, area_inventory_after_visit + ((end_time - eta)/60) * net_demand)
            deviation_after_visit = abs(ending_inventory_after_visit - target_state)
   
            ending_inventory_no_visit = max(0, initial_inventory + ((end_time - current_time)/60) * net_demand)
            deviation_no_visit = abs(ending_inventory_no_visit - target_state)

            improved_deviation = deviation_no_visit - deviation_after_visit

            # Calculate excess escooters, with and wihtout visits
            excess_escooters_after_visit = ending_inventory_after_visit
            excess_escooters_no_visit = ending_inventory_no_visit

            expected_number_of_escooters = area_inventory_after_visit
            station_type = calculate_station_type(target_state, expected_number_of_escooters)

            for neighbor in area.neighbours:
                roamings = 0
                roamings_no_visit = 0
                net_demand_neighbor = scenario_dict[neighbor.location_id]
                expected_ecooters_neighbor = len(neighbor.get_available_bikes()) + net_demand_neighbor
                neighbor_type = calculate_station_type(neighbor.get_target_state(simul.day(),simul.hour()), expected_ecooters_neighbor)

                if neighbor_type == station_type:
                    if net_demand_neighbor < 0:
                        time_first_violation = current_time + (len(neighbor.get_available_bikes())/-net_demand_neighbor) * 60
                    else:
                        time_first_violation = end_time
                    

                    if time_first_violation < end_time:
                        convertable_violations = (min(end_time - time_first_violation, end_time - eta)/60) * abs(net_demand_neighbor)

                        # Count the roamings done with and without visitation
                        if neighbor_type == 'd':
                            if convertable_violations <= excess_escooters_after_visit:
                                roamings += convertable_violations
                                excess_escooters_after_visit -= convertable_violations
                            else:
                                roamings += excess_escooters_after_visit
                                excess_escooters_after_visit -= excess_escooters_after_visit
                            
                            if convertable_violations <= excess_escooters_no_visit:
                                roamings_no_visit += convertable_violations
                                excess_escooters_no_visit -= convertable_violations
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
        TODO Area som nøkkel?
        Returns the ID of the Area with performing best on average over all the scenarios.

        Parameters:
        - vehicle = The Vehicle-object doing the action
        - simul = Simulator
        - plan_scores = dictionaries, key: Plan, value: list of float-scores for each scenario
        """
        # Make a dictionary with a value of the sum instead of a list of scores
        score_board = dict()
        for plan in plan_scores:
                score_board[plan] = sum(plan_scores[plan]) / len(plan_scores[plan]) # TODO Skal den deles slik at det blir average?

        # Sort the score board in descending order by scores
        score_board_sorted = dict(sorted(score_board.items(), key=lambda item: item[1], reverse=True))
        
        # If there is a best plan return that one
        if list(score_board_sorted.keys())[0] != None:
            best_plan = list(score_board_sorted.keys())[0]
            branch = best_plan.branch_number
            simul.metrics.add_aggregate_metric(simul, "branch"+str(branch+1), 1)
            first_move = best_plan.plan[vehicle.vehicle_id][1].station.location_id
            cluster = best_plan.plan[vehicle.vehicle_id][1].station #TODO lagres clusterne i plan?
            return first_move , cluster
        
        # If there is no best, choose a random station that is not in the tabu_list
        else:  #TODO er det potential clusters?
            tabu_list = [vehicle.location.location_id for vehicle in simul.state.get_vehicles()]
            potential_clusters = [station for station in simul.state.get_areas() if station.location_id not in tabu_list]    
            rng_balanced = np.random.default_rng(None)
            cluster = rng_balanced.choice(potential_clusters)
            return cluster.location_id, cluster        

def calculate_loading_quantities_and_swaps_greedy(vehicle, simul, cluster, overflow_criteria, starvation_criteria, swap_threshold):
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
    target_state = round(cluster.get_target_state(simul.day(), simul.hour()))
    num_max_usable_escooters_after_visit = get_max_num_usable_bikes(cluster, vehicle)

    # Count how many neighbors are starved or overflowing
    starved_neighbors = 0
    overflowing_neighbors = 0
    for neighbor in cluster.neighbours:
        num_escooters_neighbor = len(neighbor.get_available_bikes())
        neighbor_target_state = round(neighbor.get_target_state(simul.day(), simul.hour()))
        if num_escooters_neighbor < starvation_criteria * neighbor_target_state:
            starved_neighbors += 1
        elif num_escooters_neighbor > overflow_criteria * neighbor_target_state:
            overflowing_neighbors += 1

    # If the cluster is a delivery station, calculate which escooters to deliver from the vehicle, and which escooters at the cluster to swap on
    if num_max_usable_escooters_after_visit < target_state: #TODO Ta hensyn til nabocluster her?
        num_escooters_to_deliver = min(
            len([escooter for escooter in vehicle.get_ff_bike_inventory() if escooter.battery > BATTERY_LIMIT_TO_USE]), 
            target_state - num_max_usable_escooters_after_visit + BIKES_STARVED_NEIGHBOR * starved_neighbors)
        escooters_to_deliver, escooters_to_swap = get_escooter_ids_load_swap(cluster, vehicle, num_escooters_to_deliver, "deliver")
        escooters_to_pickup = []
    
    # If the cluster is a pickup station, calculate which escooters to pickup, and which to escooters at the cluster to swap batteries on
    elif num_max_usable_escooters_after_visit > target_state:
        remaining_cap_vehicle = vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory())
        number_of_escooters_to_pickup = min(
            remaining_cap_vehicle, 
            num_max_usable_escooters_after_visit - target_state + BIKES_OVERFLOW_NEIGHBOR * overflowing_neighbors, 
            len(cluster.bikes))
        escooters_to_deliver=[]
        escooters_to_pickup, escooters_to_swap = get_escooter_ids_load_swap(cluster, vehicle, number_of_escooters_to_pickup, "pickup")

    # If no escooters need to be picked up or delivered, find out how many escooters to swap batteries on
    else:
        escooters_to_pickup = []
        escooters_to_deliver = []

        escooters_in_station_low_battery = cluster.get_swappable_bikes(BATTERY_LIMIT_TO_USE)
        num_escooters_to_swap = min(
            len(escooters_in_station_low_battery),
            vehicle.battery_inventory)
        escooters_to_swap = [escooter.bike_id for escooter in escooters_in_station_low_battery[:num_escooters_to_swap]]

    # Return lists of escooter IDs to do each action on
    return escooters_to_pickup, escooters_to_deliver, escooters_to_swap

def get_max_num_usable_bikes(cluster, vehicle): 
    """"
    Returns max number of escooters in a cluster with sufficient battery level, neglects escooters that cannot be fixed.

    Parameters:
    - cluster = Cluster-object being considered
    - vehicle = Vehicle considered to rebalance station
    """
    return len(cluster.get_available_bikes()) + min(len(cluster.get_swappable_bikes(BATTERY_LIMIT_TO_USE)), vehicle.battery_inventory)

def get_escooter_ids_load_swap(cluster, vehicle, num_escooters, cluster_type):
    """
    Returns lists of the IDs of the bikes to deliver/pick-up and swap.

    Parameters:
    - cluster = Cluster being considered
    - vehicle = Vehicle doing the action
    - num_escooters = difference from target state after battery swap on site is done + effects of neighbors
    - cluster_type = if there has to be unloading or pick-ups done in cluster
    - swap_threshold = 

    TODO trenger vi å ta hensyn til FF vs SB her?
    """
    if SORTED_BIKES:
        escooters_in_station = sorted(cluster.bikes.values(), key=lambda bike: bike.battery, reverse=False)
        escooters_in_vehicle =  sorted(vehicle.get_bike_inventory(), key=lambda bike: bike.battery, reverse=False)
    else:
        escooters_in_station = list(cluster.bikes.values())
        escooters_in_vehicle =  vehicle.get_bike_inventory()

    # Returns lists of escooter IDs on which to deliver and which to swap batteries on
    if cluster_type == "deliver":
        number_of_escooters_to_swap = min(len(cluster.get_swappable_bikes(BATTERY_LIMIT_TO_USE)),vehicle.battery_inventory)
        number_of_escooters_to_deliver = int(num_escooters)+1

        escooters_to_swap = [escooter.bike_id for escooter in escooters_in_station[:number_of_escooters_to_swap]]
        escooters_to_deliver = [escooter.bike_id for escooter in escooters_in_vehicle[-number_of_escooters_to_deliver:]]

        return escooters_to_deliver, escooters_to_swap
    
    # Returns lists of escooter IDs on which to load and which to swap batteries on
    elif cluster_type == "pickup":
        # Restriction on how many bikes can be swapped and picked up
        number_of_escooters_to_swap_and_pickup = min(
            len(cluster.get_swappable_bikes(BATTERY_LIMIT_TO_SWAP)),
            vehicle.battery_inventory, 
            round(num_escooters))
        num_escooters_to_only_pickup = round(num_escooters) - number_of_escooters_to_swap_and_pickup
        num_escooters_to_only_swap = min(
            vehicle.battery_inventory, 
            max(0, len(cluster.get_swappable_bikes(BATTERY_LIMIT_TO_USE)) - number_of_escooters_to_swap_and_pickup)) if ONLY_SWAP_ALLOWED else 0

        escooters_to_swap = []
        escooters_to_pickup = [escooter.bike_id for escooter in escooters_in_station[:number_of_escooters_to_swap_and_pickup]]
        if num_escooters_to_only_pickup > 0:
            escooters_to_pickup += [escooter.bike_id for escooter in escooters_in_station[-num_escooters_to_only_pickup:]]
        
        elif num_escooters_to_only_swap > 0:
            escooters_to_swap += [escooter.bike_id for escooter in escooters_in_station[number_of_escooters_to_swap_and_pickup:number_of_escooters_to_swap_and_pickup+num_escooters_to_only_swap]]

        return escooters_to_pickup, escooters_to_swap
    
    return [],[]

def find_potential_clusters(simul, cutoff_vehicle, vehicle, bikes_in_vehicle):
    """
    TODO
    Returns a list of Station-Objects that are not in the tabu list, and that need help to reach target state.

    Parameters:
    - simul = Simulator
    - cutoff_vehicle = At what percentage is the vehicle considered "empty" or "full"
    - cutoff_station = At what percentage above or below the target state is a station considered a pickup or delivery
    - vehicle = Vehicle-object under consideration
    - bikes_at_vehicle = number of bikes in the vehicle at the time
    - tabu_list = list of Station-objects that are currently being visited or soon to be
    """
    potential_pickup_clusters = []
    potential_delivery_clusters = []

    if cutoff_vehicle * vehicle.bike_inventory_capacity <= bikes_in_vehicle <= (1-cutoff_vehicle) * vehicle.bike_inventory_capacity:
        potential_pickup_clusters = clusterPickup(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, PICKUP_CLUSTERING_THRESHOLD, MAX_WALKING_AREAS, vehicle, simul)
        potential_delivery_clusters = clusterDelivery(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, DELIVERY_CLUSTERING_THRESHOLD, MAX_WALKING_AREAS, vehicle, simul)

        potential_stations = potential_pickup_clusters + potential_delivery_clusters
        cluster_type = 'b'

    else:
        if bikes_in_vehicle <= cutoff_vehicle*vehicle.bike_inventory_capacity:
            potential_stations = clusterPickup(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, PICKUP_CLUSTERING_THRESHOLD, MAX_WALKING_AREAS, vehicle, simul)
            cluster_type = 'p'

        elif bikes_in_vehicle >= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
            potential_stations = clusterDelivery(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, DELIVERY_CLUSTERING_THRESHOLD, MAX_WALKING_AREAS, vehicle, simul)
            cluster_type = 'd'


    return potential_stations, cluster_type