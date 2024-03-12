from policies import Policy
from settings import *
import sim
from .Visit import Visit
from .Plan import Plan
from .Criticality_score import calculate_criticality, calculate_station_type
from .Simple_calculations import calculate_net_demand, copy_arr_iter, generate_discounting_factors, calculate_hourly_discharge_rate
from .dynamic_clustering import Cluster

import numpy as np
import time

class BS_PILOT(Policy):
    """
    Class for X-PILOT-BS policy. Finds the best action for e-bike systems.
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
        Returns an Action (with bike ids to swap batteries on, to pick-up, to unload, id of next location to drive to)

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
        total_num_bikes_in_system = len(simul.state.get_all_sb_bikes())

        # Goes to depot if the vehicle's battery inventory is empty on arrival, and picks up all bikes at station that is unusable
        if vehicle.battery_inventory <= 0 and len(simul.state.depots) > 0: # TODO må denne tilpasses "egne" depots
            next_location = self.simul.state.get_closest_depot(vehicle)
            bikes_to_pickup = list(vehicle.location.get_unusable_bikes().keys())
            max_pickup = min(vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()), len(bikes_to_pickup))
            return sim.Action(
                [],
                bikes_to_pickup[:max_pickup],
                [],
                next_location
            )

        # Loading and swap strategy at current location is always chosen greedily
        bikes_to_pickup, bikes_to_deliver, batteries_to_swap = calculate_loading_quantities_and_swaps_greedy(vehicle, simul, vehicle.location, self.overflow_criteria, self.starvation_criteria)
        number_of_bikes_pickup = len(bikes_to_pickup)
        number_of_bikes_deliver = len(bikes_to_deliver)
        number_of_batteries_to_swap = len(batteries_to_swap)


        # Make a plan for all vehicles
        plan_dict = dict()
        for v in simul.state.get_vehicles(): # TODO skal vi kun gå over bilene som er sb?
            # If vehicle is at a location, add current location to the plan with the greedy loading and swap strategy
            if v.eta == 0:
                plan_dict[v.vehicle_id] = [Visit(v.location, number_of_bikes_pickup, number_of_bikes_deliver, number_of_batteries_to_swap, simul.time, v)]
            
            # If the vehicle is driving, use pilot to calculate the loading and swap strategy and add to the plan
            else:
                number_of_bikes_pickup, number_of_bikes_deliver, number_of_batteries_to_swap = self.calculate_loading_quantities_and_swaps_pilot(v, simul, v.location, v.eta)
                plan_dict[v.vehicle_id] = [Visit(v.location, int(number_of_bikes_pickup), int(number_of_bikes_deliver), int(number_of_batteries_to_swap), v.eta, v)]
        
        # All locations the vehicles are at or are on their way to is added to the tabu list and plan
        tabu_list = [v.location.location_id for v in simul.state.get_vehicles()] # TODO skal vi kun gå over bilene som er sb?
        plan = Plan(plan_dict, tabu_list)

        # Use X-PILOT-BS to find which location to drive to next
        next_location = self.PILOT_function(simul, vehicle, plan, self.max_depth, self.number_of_successors, end_time, total_num_bikes_in_system)


        # Count the neighbors that are starved or congested
        similary_imbalances_starved = 0
        similary_imbalances_overflow = 0
        for neighbor in simul.state.stations[next_location].neighbours:
            if len(neighbor.get_available_bikes()) > self.overflow_criteria * neighbor.get_target_state(simul.day(),simul.hour()) and number_of_bikes_pickup > 0:
                similary_imbalances_overflow += 1
            elif len(neighbor.get_available_bikes()) < self.starvation_criteria * neighbor.get_target_state(simul.day(),simul.hour()) and number_of_bikes_deliver > 0:
                similary_imbalances_starved += 1
        
        simul.metrics.add_aggregate_metric(simul, "similarly imbalanced starved", similary_imbalances_starved)
        simul.metrics.add_aggregate_metric(simul, "similarly imbalanced congested", similary_imbalances_overflow)
        simul.metrics.add_aggregate_metric(simul, "accumulated solution time", time.time()-start_logging_time)
        simul.metrics.add_aggregate_metric(simul, 'number of problems solved', 1)

        if COLLAB_POLICY:
            current_area = vehicle.location.area
            next_area = simul.state.get_location_by_id(next_location).area

            current_cluster = Cluster([current_area], current_area, current_area.bikes, current_area.neighbours)
            next_cluster = Cluster([next_area], next_area, next_area.bikes, next_area.neighbours)

            # Add neighboring areas as far as the radius allows
            for _ in range(OPERATOR_AREA_RADIUS): # TODO legg til max radius for operatør
                current_cluster.areas.extend(current_cluster.neighbours)
                current_cluster.bikes.update({bike.bike_id: bike
                                      for area in current_cluster.neighbours
                                      for bike in area.get_bikes()
                                      })
                current_cluster.neighbours = list(set([neighbor 
                                      for area in current_cluster.neighbours 
                                      for neighbor in area.neighbours 
                                      if neighbor not in current_cluster.areas]))
                
                next_cluster.areas.extend(next_cluster.neighbours)
                next_cluster.bikes.update({bike.bike_id: bike
                                      for area in next_cluster.neighbours
                                      for bike in area.get_bikes()
                                      })
                next_cluster.neighbours = list(set([neighbor 
                                      for area in next_cluster.neighbours 
                                      for neighbor in area.neighbours 
                                      if neighbor not in next_cluster.areas]))

            current_area_inventory = len(current_cluster.get_available_bikes())
            current_area_target_state = current_cluster.get_target_state(simul.day(), simul.day())

            # TODO burde disse vurdere hvordan det ser ut i fremtiden?
            next_area_inventory = len(next_cluster.get_available_bikes())
            next_area_target_state = next_cluster.get_target_state(simul.day(), simul.day())

            num_current_area_overflow = current_area_inventory - current_area_target_state
            num_next_area_lacking = next_area_inventory - next_area_target_state
            
            if num_current_area_overflow > 0 and num_next_area_lacking > 0: # Hjelper kun til hvis det er for mange sykler der
                left_bike_capacity = vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()) - len(bikes_to_pickup)
                num_escooters_to_pickup = min(left_bike_capacity, num_current_area_overflow, num_next_area_lacking)
                escooters_to_pickup = [] # TODO bruke num_escooter_to_pick_up til å finne hvilke escootere som skal hentes og skriv det i en liste med id-en

                return sim.Action(
                    batteries_to_swap,
                    bikes_to_pickup,
                    bikes_to_deliver,
                    next_location,
                    helping_pickup = escooters_to_pickup
                )

        return sim.Action(
            batteries_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_location
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
        - total_num_bikes_in_system = Total station based number of bikes
        """

        # Create a tree of possible plans, each are evaluated with different criticality weights
        completed_plans = []
        for weight_set in self.criticality_weights_set:
            plans = [[] for i in range(max_depth+1)]
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
                        num_successors_other_vehicle = max(1, number_of_successors//2)
                        new_visits = self.greedy_next_visit(plan, simul, num_successors_other_vehicle, weight_set, total_num_bikes_in_system)
                    else:
                        new_visits = self.greedy_next_visit(plan, simul, number_of_successors, weight_set, total_num_bikes_in_system)
                    
                    # If there are no new visits or there is no visit within the time frame, finalize the plan
                    if new_visits == None or plan.next_visit.get_depature_time() > end_time:
                        new_plan = Plan(plan.copy_plan(), copy_arr_iter(plan.tabu_list), weight_set, plan.branch_number)
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
        
        # Returns the location with the best average score over all scenarios
        return self.return_best_move_average(vehicle, simul, plan_scores)

    def calculate_loading_quantities_and_swaps_pilot(self, vehicle, simul, station, eta):
        """
        Returns the NUMBER of bikes to pick up, deliver and swap batteries on. Takes future demand into account by calculating it for the next hour, and treats it as evenly distributed throughout that hour

        Parameters:
        - vehicle = Vehicle-object the action is considered for
        - simul = Simulator
        - station = Station the vehicle is considering doing the action at
        - eta = Estimated time of arrival for the vehicle to arrive at station
        """
        num_vehicle_bike_inventory = len(vehicle.get_bike_inventory())
        number_of_bikes_pickup = 0
        number_of_bikes_deliver = 0
        number_of_battery_swaps = 0

        target_state = round(station.get_target_state(simul.day(),simul.hour())) 
        net_demand = calculate_net_demand(station, simul.time, simul.day(), simul.hour(), 60)
        max_num_usable_bikes = get_max_num_usable_bikes(station, vehicle)
        max_num_usable_bikes_eta = max_num_usable_bikes + ((eta - simul.time)/60)*net_demand # How many bikes at station at eta, based on demand forecast

        # Calculate the amount of neighbors that are starving or congested, can impact the number of bikes to operate on
        num_starved_neighbors = 0
        num_overflowing_neighbors = 0
        for neighbor in station.neighbours:
            net_demand_neighbor = calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(),60)
            num_usable_bikes_neighbor_eta = len(neighbor.get_available_bikes()) + ((eta - simul.time)/60)*net_demand_neighbor
            target_state_neighbor = round(neighbor.get_target_state(simul.day(), simul.hour()))
            if num_usable_bikes_neighbor_eta < self.starvation_criteria * target_state_neighbor:
                num_starved_neighbors += 1
            elif num_usable_bikes_neighbor_eta > self.overflow_criteria * target_state_neighbor:
                num_overflowing_neighbors += 1

        # Calculate how many bikes to do different actions on
        if max_num_usable_bikes_eta < target_state: # deliver
            number_of_bikes_deliver = int(min(num_vehicle_bike_inventory, target_state - max_num_usable_bikes_eta + BIKES_STARVED_NEIGHBOR * num_starved_neighbors)) + 1 # round up
            number_of_battery_swaps = min( len(station.get_swappable_bikes(BATTERY_LIMIT_TO_USE)), vehicle.battery_inventory)
        elif max_num_usable_bikes_eta > target_state: # pick-up
            remaining_cap_vehicle = vehicle.bike_inventory_capacity - num_vehicle_bike_inventory
            num_less_bikes = min(remaining_cap_vehicle, max_num_usable_bikes_eta - target_state + BIKES_OVERFLOW_NEIGHBOR * num_overflowing_neighbors)

            num_bikes_to_swap_and_pickup = min(len(station.get_swappable_bikes(BATTERY_LIMIT_TO_SWAP)), vehicle.battery_inventory, round(num_less_bikes))
            number_of_bikes_pickup = round(num_less_bikes) - num_bikes_to_swap_and_pickup
            number_of_battery_swaps = min(vehicle.battery_inventory, max(0,len(station.get_swappable_bikes(BATTERY_LEVEL_LOWER_BOUND)) - num_bikes_to_swap_and_pickup)) # TODO skal ONLY_SWAP brukes?
        else: # only swap
            unusable_bikes = station.get_unusable_bikes()
            number_of_battery_swaps = min(len(unusable_bikes), vehicle.battery_inventory)
        
        return number_of_bikes_pickup, number_of_bikes_deliver, number_of_battery_swaps
    


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
        num_bikes_now = len(vehicle.get_bike_inventory())

        # Update the vehicle bike inventory based on the planned operational actions
        for visit in plan.plan[vehicle.vehicle_id]:
            num_bikes_now += visit.loading_quantity
            num_bikes_now -= visit.unloading_quantity

        # Finds potential next stations based on pick up or delivery status of the station and tabulist
        potential_stations = find_potential_stations(simul, VEHICLE_TYPE_MARGIN, LOCATION_TYPE_MARGIN, vehicle, num_bikes_now, tabu_list) # TODO fiks 0.15, hva betyr det? Margin greiene
        if potential_stations == []:
            return None
        
        number_of_successors = min(number_of_successors, len(potential_stations))

        # Finds the criticality score of all potential stations, and sort them in descending order
        stations_sorted = calculate_criticality(weight_set, simul, potential_stations, plan.plan[vehicle.vehicle_id][-1].station, total_num_bikes_in_system ,tabu_list) # TODO fjerne station_type?
        stations_sorted_list = list(stations_sorted.keys())

        # Selects the most critical stations as next visits
        next_stations = stations_sorted_list[:number_of_successors]

        for next_station in next_stations:
            arrival_time = plan.plan[vehicle.vehicle_id][-1].get_depature_time() + simul.state.traveltime_vehicle_matrix[(plan.plan[vehicle.vehicle_id][-1].station.location_id, next_station.location_id)] + MINUTES_CONSTANT_PER_ACTION
            num_bikes_to_pickup, num_bikes_to_deliver, num_bikes_to_swap = self.calculate_loading_quantities_and_swaps_pilot(vehicle, simul, next_station, arrival_time)
            new_visit = Visit(next_station, num_bikes_to_pickup, num_bikes_to_deliver, num_bikes_to_swap, arrival_time, vehicle)
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
        locations_dict = simul.state.get_sb_locations()
        if number_of_scenarios < 1:
            scenario_dict = dict() 
            for station_id in locations_dict:
                net_demand =  calculate_net_demand(locations_dict[station_id], simul.time ,simul.day(),simul.hour(), TIME_HORIZON) #returns net demand for next hour 
                scenario_dict[station_id] = net_demand
            scenarios.append(scenario_dict)
        
        else:
            for s in range(number_of_scenarios):
                scenario_dict = dict()
                planning_horizon = TIME_HORIZON
                time_now = simul.time
                day = simul.day()
                hour = simul.hour()
                minute_in_current_hour = time_now - day*24*60 - hour*60
                minutes_current_hour = min(60 - minute_in_current_hour, planning_horizon)
                minutes_next_hour = planning_horizon - minutes_current_hour
                
                # TODO gange med 2?
                for station_id in locations_dict: 
                    expected_arrive_intensity = 2*locations_dict[station_id].get_arrive_intensity(simul.day(), simul.hour())
                    expected_leave_intensity = 2*locations_dict[station_id].get_leave_intensity(simul.day(), simul.hour())
                    expected_arrive_intensity_next = 2*locations_dict[station_id].get_arrive_intensity(simul.day(), simul.hour()+1)
                    expected_leave_intensity_next = 2*locations_dict[station_id].get_leave_intensity(simul.day(), simul.hour()+1)
                    
                    if poisson:
                        net_demand_current = rng.poisson(expected_arrive_intensity) - rng.poisson(expected_leave_intensity)
                        net_demand_next = rng.poisson(expected_arrive_intensity_next) - rng.poisson(expected_leave_intensity_next)
                        net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/planning_horizon
                    
                    else: #normal_dist
                        arrive_intensity_stdev = locations_dict[station_id].get_arrive_intensity_stdev(simul.day(), simul.hour())
                        leave_intensity_stdev = locations_dict[station_id].get_leave_intensity_stdev(simul.day(), simul.hour())
                        arrive_intensity_stdev_next = locations_dict[station_id].get_arrive_intensity_stdev(simul.day(), simul.hour()+1)
                        leave_intensity_stdev_next = locations_dict[station_id].get_leave_intensity_stdev(simul.day(), simul.hour()+1)

                        net_demand_current = rng.normal(expected_arrive_intensity, arrive_intensity_stdev) - rng.normal(expected_leave_intensity, leave_intensity_stdev)
                        net_demand_next = rng.normal(expected_arrive_intensity_next, arrive_intensity_stdev_next) - rng.normal(expected_leave_intensity_next, leave_intensity_stdev_next)
                        net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/planning_horizon
                        
                    scenario_dict[station_id] = net_demand 
                scenarios.append(scenario_dict)
        return scenarios
    
    def evaluate_route(self, route, scenario_dict, end_time, simul, weights, total_num_bikes_in_system):
        """
        Returns the score based on if the vehicle drives this route in comparisson to not driving it at all

        Parameters:
        - route = list of visits the vehicle is supposed to do
        - scenario_dict = a dictionary with a possible net demand for each station
        - end_time = the stopping point of the horizon to evaluate
        - simul = Simulator
        - weights = weights for avoided violations, neighbor roamings, and improved deviation
        - total_num_bikes_in_system = the total amount of bicycles that are in the SB system
        """
        
        discounting_factors = generate_discounting_factors(len(route), self.discounting_factor)
        avoided_disutility = 0
        current_time = simul.time
        counter = 0

        for visit in route:
            avoided_violations = 0
            neighbor_roamings = 0
            improved_deviation = 0

            station = visit.station

            loading_quantity = visit.loading_quantity
            unloading_quantity = visit.unloading_quantity
            swap_quantity = visit.swap_quantity

            neighbors = station.neighbours

            eta = visit.arrival_time

            if eta > end_time:
                eta = end_time
            
            initial_inventory = len(station.get_available_bikes())
            net_demand = scenario_dict[station.location_id]
            target_state = station.get_target_state(simul.day(), simul.hour())

            #########################################################################
            # AVOIDED VIOLATIONS                                                    #
            # Below we implement all the ways we want to measure avoided violations #
            # I have removed conjuctions as a violation                             #
            #########################################################################

            if net_demand < 0:
                sorted_escooters_in_station = sorted(station.bikes.values(), key=lambda bike: bike.battery, reverse=False)
                time_first_violation_no_visit = current_time + min((station.number_of_bikes() - len(station.get_swappable_bikes(BATTERY_LEVEL_LOWER_BOUND)))/ -net_demand, (sum(Ebike.battery for Ebike in sorted_escooters_in_station[-3:])/3)/(calculate_hourly_discharge_rate(simul, total_num_bikes_in_system)*60))
            else:
                time_first_violation_no_visit = end_time
            
            # Number of violations that happen if we dont visit the station
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
            
            # Number of escooters at station after visit
            inventory_after_loading_and_swaps = initial_inventory + ((eta - current_time)/60)*net_demand - unavoidable_violations - loading_quantity + unloading_quantity + swap_quantity
            
            # Time for first violation if we visit
            if net_demand < 0:
                if swap_quantity > loading_quantity+2:
                    time_first_violation_after_visit = eta + min((inventory_after_loading_and_swaps/(-net_demand))*60, 100/calculate_hourly_discharge_rate(simul, total_num_bikes_in_system))
                else:
                    time_first_violation_after_visit = eta + min((inventory_after_loading_and_swaps/(-net_demand))*60, (sum(ebike.battery for ebike in sorted_escooters_in_station[-3:])/3)/(calculate_hourly_discharge_rate(simul, total_num_bikes_in_system)*60))
            else:
                time_first_violation_after_visit = end_time
            
            if time_first_violation_after_visit < end_time:
                violations_after_visit = ((end_time - time_first_violation_after_visit)/60) * net_demand
            else:
                violations_after_visit = 0

            
            avoided_violations = violation_no_visit - violations_after_visit

            
            #############################################
            # IMPROVED DEVIATION                        #
            # Deviation from target station after visit #
            #############################################
    
            ending_inventory = max(0, inventory_after_loading_and_swaps + ((end_time - eta)/60) * net_demand)
            deviation_visit = abs(ending_inventory - target_state)
   
            ending_inventory_no_visit = max(0, initial_inventory + ((end_time - current_time)/60) * net_demand)
            deviation_no_visit = abs(ending_inventory_no_visit - target_state)

            improved_deviation = deviation_no_visit - deviation_visit


            #################################################################
            # NEIGHBOR ROAMINGS                                             #
            # How much unmet demand at other clusters can be cought by this #
            # Removed evaluation for roaming for locks                      #
            #################################################################

            excess_escooters = ending_inventory
            excess_escooters_no_visit = ending_inventory_no_visit

            expected_number_of_escooters = inventory_after_loading_and_swaps
            station_type = calculate_station_type(target_state, expected_number_of_escooters)

            for neighbor in neighbors:
                roamings = 0
                roamings_no_visit = 0
                net_demand_neighbor = scenario_dict[neighbor.location_id]
                expected_ecooters_neighbor = neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(BATTERY_LEVEL_LOWER_BOUND)) + net_demand_neighbor
                neighbor_type = calculate_station_type(neighbor.get_target_state(simul.day(),simul.hour()),expected_ecooters_neighbor)

                if neighbor_type == station_type:
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
                        
            
                distance_scaling = ((simul.state.get_vehicle_travel_time(station.location_id, neighbor.location_id)/60)* VEHICLE_SPEED)/MAX_ROAMING_DISTANCE_SOLUTIONS
                neighbor_roamings += (1-distance_scaling)*roamings-roamings_no_visit
            
            avoided_disutility += discounting_factors[counter]*(weights[0]*avoided_violations + weights[1]*neighbor_roamings + weights[2]*improved_deviation)

            counter += 1
        
        return avoided_disutility

    ##########################################################################
    # Finds the action that performs best over the most scenarios            #
    # TODO fix if we are going to use this                                   #
    ##########################################################################    

    def return_best_move(self, vehicle, simul, plan_scores): #returns station_id 
        score_board = dict() #station id : number of times this first move returns the best solution
        num_scenarios=self.number_of_scenarios
        if num_scenarios==0:
            num_scenarios+=1 #this scenario is now the expected value 
        for scenario_id in range(num_scenarios):
            best_score = -1000
            best_plan = None
            for plan in plan_scores:
                if plan_scores[plan][scenario_id] > best_score:
                    best_plan = plan
                    best_score = plan_scores[plan][scenario_id]
            
            if best_plan == None:
                tabu_list = [vehicle2.location.location_id for vehicle2 in simul.state.get_vehicles()]
                potential_stations2 = [station for station in simul.state.get_stations() if station.location_id not in tabu_list]    
                rng_balanced = np.random.default_rng(None)
                print("lunsj!")
                return rng_balanced.choice(potential_stations2).location_id 

            best_first_move = best_plan.plan[vehicle.vehicle_id][1].station.location_id
            if best_first_move in score_board:
                score_board[best_first_move] += 1 
            else:
                score_board[best_first_move] = 1 

            simul.metrics.add_aggregate_metric(simul, "branch"+str(best_plan.branch_number+1), 1)
            simul.metrics.add_aggregate_metric(simul, "weight_set"+str(best_plan.weight_set), 1)
           
        score_board_sorted = dict(sorted(score_board.items(), key=lambda item: item[1], reverse=True))

        return list(score_board_sorted.keys())[0]

    ##########################################################################
    # Finds the action which on avarage performs best over several scenarios #
    ##########################################################################
    def return_best_move_average(self, vehicle, simul, plan_scores):
        score_board = dict()
        num_scenarios = self.number_of_scenarios
        if num_scenarios == 0:
            num_scenarios += 1
        for scenario_id in range(num_scenarios):
            for plan in plan_scores:
                score = plan_scores[plan][scenario_id]
                if plan in score_board:
                    score_board[plan] += score
                else:
                    score_board[plan] = score

        score_board_sorted = dict(sorted(score_board.items(), key=lambda item: item[1], reverse=True))
        if list(score_board_sorted.keys())[0] != None:
            best_plan = list(score_board_sorted.keys())[0]
            branch = best_plan.branch_number
            simul.metrics.add_aggregate_metric(simul, "branch"+str(branch+1), 1)
            first_move = best_plan.plan[vehicle.vehicle_id][1].station.location_id
            return first_move
        else: 
            tabu_list = [vehicle2.location.location_id for vehicle2 in simul.state.get_vehicles()]
            potential_stations2 = [station for station in simul.state.get_stations() if station.location_id not in tabu_list]    
            rng_balanced = np.random.default_rng(None)
            return rng_balanced.choice(potential_stations2).location_id

#############################################################################################
#   Number of bikes to pick up / deliver is choosen greedy based on clusters in reach       #
#   Which bike ID´s to pick up / deliver is choosen based on battery level                  #
#   How many and which escooters to swap battery on based on inventory and station status   #                                      
#   Applied functionality such that scooters with battery level < thershold does not count  #             
#############################################################################################

def calculate_loading_quantities_and_swaps_greedy(vehicle, simul, station, overflow_criteria, starvation_criteria):
    num_escooters_vehicle = len(vehicle.get_bike_inventory())

    target_state = round(station.get_target_state(simul.day(), simul.hour())) #Denne må vi finne ut hvordan lages
    num_escooters_station = station.number_of_bikes() # number of scooters at the station
    num_escooters_accounted_for_battery_swaps = get_max_num_usable_bikes(station, vehicle) # 

    ################################################################
    #  Adjusting numbers based on status at neighboring stations   #
    #  Changed from capacity to diviation from ideal state         #
    ################################################################

    starved_neighbors = 0
    overflowing_neighbors = 0

    for neighbor in station.neighbours:
        num_escooters_neighbor = neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(BATTERY_LEVEL_LOWER_BOUND))
        neighbor_target_state = round(neighbor.get_target_state(simul.day(), simul.hour()))
        if num_escooters_neighbor < starvation_criteria * neighbor_target_state:
            starved_neighbors += 1
        elif num_escooters_neighbor > overflow_criteria * neighbor_target_state:
            overflowing_neighbors += 1

    #######################################################################
    #  Find out whether the station is a pick-up or a deliver station     #
    #  And based on that the quantities for pickup, deliveries and swaps  #
    #######################################################################

    if num_escooters_accounted_for_battery_swaps < target_state: #Ta hensyn til nabocluster her?
        number_of_escooters_to_deliver = min(len([escooter for escooter in vehicle.get_bike_inventory() if escooter.battery > BATTERY_LEVEL_LOWER_BOUND]), target_state - num_escooters_accounted_for_battery_swaps + BIKES_STARVED_NEIGHBOR * starved_neighbors) # discuss 2*starved_neighbors part, ta hensyn til postensielle utladede scootere i bilen
        escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = get_bike_ids_load_swap(station, vehicle, number_of_escooters_to_deliver, "deliver")
        escooters_to_pickup_accounted_for_battery_swaps = []
    
    elif num_escooters_accounted_for_battery_swaps > target_state:
        remaining_cap_vehicle = vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory())
        number_of_escooters_to_pickup = min(remaining_cap_vehicle, num_escooters_accounted_for_battery_swaps - target_state + BIKES_OVERFLOW_NEIGHBOR * overflowing_neighbors, len(station.bikes)) #discuss logic behind this
        escooters_to_deliver_accounted_for_battery_swaps=[]
        escooters_to_pickup_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = get_bike_ids_load_swap(station, vehicle, number_of_escooters_to_pickup, "pickup")

    else:
        escooters_to_pickup_accounted_for_battery_swaps = []
        escooters_to_deliver_accounted_for_battery_swaps = []

        escooters_in_station_low_battery = station.get_swappable_bikes(BATTERY_LEVEL_LOWER_BOUND)
        num_escooters_to_swap = min(len(escooters_in_station_low_battery),vehicle.battery_inventory)
        escooters_to_swap_accounted_for_battery_swap = [escooter.bike_id for escooter in escooters_in_station_low_battery[:num_escooters_to_swap]]


    
    return escooters_to_pickup_accounted_for_battery_swaps, escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap




#######################################################################################################################
# Simple calculation function to take low battery level into consideration when choosing number to deliver or pickup  #
#######################################################################################################################

def get_max_num_usable_bikes(station, vehicle): 
    """"
    Returns max number of bikes at station with sufficient battery level, neglects bikes that cannot be fixed.

    Parameters:
    - station = Station being considered
    - vehicle = Vehicle considered to rebalance station
    """
    return len(station.get_available_bikes()) + min(len(station.get_swappable_bikes()), vehicle.battery_inventory)

############################################################################################
# If deliver, swap as many low battries in station as possible then deliver rest of bikes  #
# Swaps from low battry until threshold, default 20                                        #
# Delivers from high battery until threshold                                               #  
#                                                                                          #
# If pickup, pickup and swap as many escooters with low battry as possible                 #
# Then pickup rest from high battery end or swap remaining swaps                           #
############################################################################################

def get_bike_ids_load_swap(station, vehicle, num_bikes, station_type):
    """
    Returns lists of the IDs of the bikes to deliver/pick-up and swap.

    Parameters:
    - station = Station being considered
    - vehicle = Vehicle doing the action
    - num_bikes = difference from target state after battery swap on site is done + effects of neighbors
    - station_type = if there has to be unloading or pick-ups done at the station
    """
    if SORTED_BIKES:  # sorted from lowest to highest bettery level
        bikes_at_station = sorted(station.bikes.values(), key=lambda bike: bike.battery, reverse=False) 
        vehicle_bikes =  sorted(vehicle.get_bike_inventory(), key=lambda bike: bike.battery, reverse=False)
    else:
        bikes_at_station = list(station.bikes.values())
        vehicle_bikes =  vehicle.get_bike_inventory()

    # Returns lists of bike IDs on which to deliver and which to swap batteries on
    if station_type == "deliver":
        num_bikes_to_swap = min( len(station.get_swappable_bikes(BATTERY_LIMIT_TO_USE)), vehicle.battery_inventory)
        num_bikes_to_deliver = int(num_bikes)+1

        bikes_to_swap = [bike.bike_id for bike in bikes_at_station[:num_bikes_to_swap]] # swap the n escooters with lowest percentage at the station
        bikes_to_deliver = [bike.bike_id for bike in vehicle_bikes[-num_bikes_to_deliver:]] # unload the n escooters with the highest percentage

        return bikes_to_deliver, bikes_to_swap
    
    # Returns lists of bike IDs on which to load and which to swap batteries on
    elif station_type == "pickup":
        # Restriction on how many bikes can be swapped and picked up
        num_bikes_to_swap_and_pickup = min(
            len(station.get_swappable_bikes(BATTERY_LIMIT_TO_SWAP)),
            vehicle.battery_inventory, 
            round(num_bikes)
            )
        num_bikes_to_only_pickup = round(num_bikes) - num_bikes_to_swap_and_pickup
        num_bikes_to_only_swap = min(vehicle.battery_inventory, max(0,len(station.get_swappable_bikes(BATTERY_LEVEL_LOWER_BOUND)) - num_bikes_to_swap_and_pickup)) if ONLY_SWAP_ALLOWED else 0

        bikes_to_swap = []
        bikes_to_pickup = [bike.bike_id for bike in bikes_at_station[:num_bikes_to_swap_and_pickup]]
        if num_bikes_to_only_pickup > 0:
            bikes_to_pickup += [bike.bike_id for bike in bikes_at_station[-num_bikes_to_only_pickup:]]
        
        elif num_bikes_to_only_swap > 0:
            bikes_to_swap += [bike.bike_id for bike in bikes_at_station[num_bikes_to_swap_and_pickup:num_bikes_to_swap_and_pickup+num_bikes_to_only_swap]]

        return bikes_to_pickup, bikes_to_swap
    
    return [],[]

def find_potential_stations(simul, cutoff_vehicle, cutoff_station, vehicle, bikes_at_vehicle, tabu_list):
    """
    
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
    net_demands = {station.location_id: calculate_net_demand(station, simul.time, simul.day(), simul.hour(), TIME_HORIZON) for station in potential_stations}
    target_states = {station.location_id: station.get_target_state(simul.day(), simul.hour()) for station in potential_stations}
    
    # If the available bikes in the future is bigger that a cutoff percentage of target state, the station is a pickup station
    potential_pickup_stations = [station for station in potential_stations
                                 if get_max_num_usable_bikes(station, vehicle) + net_demands[station.location_id] > (1 + cutoff_station) * target_states[station.location_id]
                                 ] # TODO skal vi bruke get_max_num_bikes?
    
    # If the available bikes after a visit in the future is lower that a cutoff percentage of target state, the station is a delivery station
    potential_delivery_stations = [ station for station in potential_stations
                                 if get_max_num_usable_bikes(station, vehicle) + net_demands[station.location_id] < (1 - cutoff_station) * target_states[station.location_id]
                                 ]
    
    #Decides pickup, delivery or both is relevant for 
    if cutoff_vehicle * vehicle.bike_inventory_capacity <= bikes_at_vehicle <= (1 - cutoff_vehicle) * vehicle.bike_inventory_capacity:
        potential_stations = potential_pickup_stations + potential_delivery_stations

    else:
        if bikes_at_vehicle <= cutoff_vehicle * vehicle.bike_inventory_capacity:
            potential_stations = potential_pickup_stations

        elif bikes_at_vehicle >= (1 - cutoff_vehicle) * vehicle.bike_inventory_capacity:
            potential_stations = potential_delivery_stations
    
    return potential_stations