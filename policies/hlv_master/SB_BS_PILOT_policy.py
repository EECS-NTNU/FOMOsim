from policies import Policy
from settings import *
import sim
from .Visit import Visit
from .Plan import Plan
from .SB_Criticality_score import calculate_criticality, calculate_station_type
from .Simple_calculations import calculate_net_demand, copy_arr_iter, generate_discounting_factors, calculate_hourly_discharge_rate

import numpy as np
import time

class BS_PILOT(Policy):
    """
    Class for X-PILOT-BS policy. Finds the best action for station based bike systems.
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

    def get_best_action(self, simul: sim.Simulator, vehicle: sim.Vehicle):
        """
        Returns an Action (with which bikes to swap batteries on, which bikes to pick-up, which bikes to unload, next location to drive to)

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
        if vehicle.battery_inventory <= 0 and len(simul.state.get_depots()) > 0: # TODO
            next_location = self.simul.state.get_closest_depot(vehicle)
            bikes_to_pickup = vehicle.location.get_unusable_bikes()
            max_pickup = min(vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()), len(bikes_to_pickup))
            return sim.Action(
                [],
                bikes_to_pickup[:max_pickup],
                [],
                next_location
            )

        start_time = time.time()
        # Loading and swap strategy at current location is always chosen greedily
        bikes_to_pickup, bikes_to_deliver, batteries_to_swap = calculate_loading_quantities_and_swaps_greedy(vehicle, simul, vehicle.location, self.overflow_criteria, self.starvation_criteria)
        number_of_bikes_pickup = len(bikes_to_pickup)
        number_of_bikes_deliver = len(bikes_to_deliver)
        number_of_batteries_to_swap = len(batteries_to_swap)
        print("time to find action SB:", time.time() - start_time)

        start_time = time.time()
        # Make a plan for all vehicles
        plan_dict = dict()
        for v in simul.state.get_sb_vehicles():
            # If vehicle is at a location, add current location to the plan with the greedy loading and swap strategy
            if v.eta == 0:
                plan_dict[v.vehicle_id] = [Visit(v.location, number_of_bikes_pickup, number_of_bikes_deliver, number_of_batteries_to_swap, simul.time, v)]
            
            # If the vehicle is driving, use pilot to calculate the loading and swap strategy and add to the plan
            else:
                number_of_bikes_pickup, number_of_bikes_deliver, number_of_batteries_to_swap = self.calculate_loading_quantities_and_swaps_pilot(v, simul, v.location, v.eta)
                plan_dict[v.vehicle_id] = [Visit(v.location, int(number_of_bikes_pickup), int(number_of_bikes_deliver), int(number_of_batteries_to_swap), v.eta, v)]
        
        # All locations the vehicles are at or are on their way to is added to the tabu list and plan
        tabu_list = [v.location.location_id for v in simul.state.get_sb_vehicles()]
        plan = Plan(plan_dict, tabu_list)

        # Use X-PILOT-BS to find which location to drive to next
        next_location = self.PILOT_function(simul, vehicle, plan, self.max_depth, self.number_of_successors, end_time, total_num_bikes_in_system)
        print("time to find next loc SB:", time.time() - start_time)

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
        num_vehicle_bike_inventory = len(vehicle.get_sb_bike_inventory())
        number_of_bikes_pickup = 0
        number_of_bikes_deliver = 0
        number_of_battery_swaps = 0

        target_state = round(station.get_target_state(simul.day(),simul.hour())) 
        time_until_arrival = eta - simul.time
        net_demand = calculate_net_demand(station, simul.time, simul.day(), simul.hour(), min(time_until_arrival, 60))
        max_num_usable_bikes = get_max_num_usable_bikes(station, vehicle)
        max_num_usable_bikes_eta = max_num_usable_bikes + net_demand

        # Calculate the amount of neighbors that are starving or congested, can impact the number of bikes to operate on
        num_starved_neighbors = 0
        num_overflowing_neighbors = 0
        for neighbor in station.neighbours:
            net_demand_neighbor = calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(), min(time_until_arrival, 60))
            num_usable_bikes_neighbor_eta = len(neighbor.get_available_bikes()) + net_demand_neighbor
            target_state_neighbor = round(neighbor.get_target_state(simul.day(), simul.hour()))
            if num_usable_bikes_neighbor_eta < self.starvation_criteria * target_state_neighbor:
                num_starved_neighbors += 1
            elif num_usable_bikes_neighbor_eta > self.overflow_criteria * target_state_neighbor:
                num_overflowing_neighbors += 1

        # Calculate how many bikes to do different actions on
        if max_num_usable_bikes_eta < target_state: # deliver
            number_of_bikes_deliver = int(min(num_vehicle_bike_inventory, target_state - max_num_usable_bikes_eta + BIKES_STARVED_NEIGHBOR * num_starved_neighbors)) + 1 # TODO ???
            number_of_battery_swaps = min( len(station.get_swappable_bikes(BATTERY_LIMIT_TO_USE)), vehicle.battery_inventory)
        elif max_num_usable_bikes_eta > target_state: # pick-up
            remaining_cap_vehicle = vehicle.bike_inventory_capacity - num_vehicle_bike_inventory
            num_less_bikes = min(remaining_cap_vehicle, max_num_usable_bikes_eta - target_state + BIKES_OVERFLOW_NEIGHBOR * num_overflowing_neighbors)

            num_bikes_to_swap_and_pickup = min(len(station.get_swappable_bikes(BATTERY_LIMIT_TO_SWAP)), vehicle.battery_inventory, round(num_less_bikes))
            number_of_bikes_pickup = round(num_less_bikes) - num_bikes_to_swap_and_pickup
            number_of_battery_swaps = min(vehicle.battery_inventory, max(0,len(station.get_swappable_bikes(BATTERY_LIMIT_TO_USE)) - num_bikes_to_swap_and_pickup))
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
        num_bikes_now = len(vehicle.get_sb_bike_inventory())

        # Update the vehicle bike inventory based on the planned operational actions
        for visit in plan.plan[vehicle.vehicle_id]:
            num_bikes_now += visit.loading_quantity
            num_bikes_now -= visit.unloading_quantity

        # Finds potential next stations based on pick up or delivery status of the station and tabulist
        potential_stations = find_potential_stations(simul,0.15,0.15,vehicle, num_bikes_now, tabu_list) # TODO margin
        if potential_stations == []:
            print("No potential stations")
            if len(plan.plan[vehicle.vehicle_id]) == 1:
                print(plan.plan[vehicle.vehicle_id])
            return None
        
        number_of_successors = min(number_of_successors, len(potential_stations))

        # Finds the criticality score of all potential stations, and sort them in descending order
        stations_sorted = calculate_criticality(weight_set, simul, potential_stations, plan.plan[vehicle.vehicle_id][-1].station, total_num_bikes_in_system ,tabu_list)
        stations_sorted_list = list(stations_sorted.keys())

        # Selects the most critical stations as next visits
        next_stations = stations_sorted_list[:number_of_successors]

        for next_station in next_stations:
            arrival_time = plan.plan[vehicle.vehicle_id][-1].get_depature_time() + simul.state.get_vehicle_travel_time(plan.plan[vehicle.vehicle_id][-1].station.location_id, next_station.location_id) + MINUTES_CONSTANT_PER_ACTION
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
                net_demand =  calculate_net_demand(locations_dict[station_id], simul.time ,simul.day(),simul.hour(), 60)
                scenario_dict[station_id] = net_demand
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

                # Make a dictionary deciding the expected net_nemand for each station
                # key = station_id, value = net_demand (either decided through poisson or normal distribution)
                for station_id in locations_dict: 
                    expected_arrive_intensity = locations_dict[station_id].get_arrive_intensity(day, hour)
                    expected_leave_intensity = locations_dict[station_id].get_leave_intensity(day, hour)
                    expected_arrive_intensity_next = locations_dict[station_id].get_arrive_intensity(next_day, next_hour)
                    expected_leave_intensity_next = locations_dict[station_id].get_leave_intensity(next_day, next_hour)
                    
                    if poisson:
                        net_demand_current = rng.poisson(expected_arrive_intensity) - rng.poisson(expected_leave_intensity)
                        net_demand_next = rng.poisson(expected_arrive_intensity_next) - rng.poisson(expected_leave_intensity_next)
                        net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/planning_horizon
                    
                    else: #normal_dist
                        arrive_intensity_stdev = locations_dict[station_id].get_arrive_intensity_stdev(day, hour)
                        leave_intensity_stdev = locations_dict[station_id].get_leave_intensity_stdev(day, hour)
                        arrive_intensity_stdev_next = locations_dict[station_id].get_arrive_intensity_stdev(next_day, next_hour)
                        leave_intensity_stdev_next = locations_dict[station_id].get_leave_intensity_stdev(next_day, next_hour)

                        net_demand_current = rng.normal(expected_arrive_intensity, arrive_intensity_stdev) - rng.normal(expected_leave_intensity, leave_intensity_stdev)
                        net_demand_next = rng.normal(expected_arrive_intensity_next, arrive_intensity_stdev_next) - rng.normal(expected_leave_intensity_next, leave_intensity_stdev_next)
                        net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/planning_horizon
                        
                    scenario_dict[station_id] = net_demand 
                scenarios.append(scenario_dict)
        
        # Return a list of num_scenarios dictionaries with expected net demand for each station in the future
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
        counter = 0 # which stage during the visit the vehicle is at -> used to discount the score

        for visit in route:
            avoided_violations = 0
            neighbor_roamings = 0
            improved_deviation = 0

            loading_quantity = visit.loading_quantity
            unloading_quantity = visit.unloading_quantity
            swap_quantity = visit.swap_quantity

            station = visit.station
            neighbors = station.neighbours

            eta = visit.arrival_time

            if eta > end_time:
                eta = end_time
            
            initial_inventory = len(station.get_available_bikes()) # TODO congestion behandles annerledes
            net_demand = scenario_dict[station.location_id]
            target_state = station.get_target_state(simul.day(), simul.hour())

            # Calculate when the first starvation or congestion will occur if not visited
            if net_demand < 0:
                sorted_bikes_in_station = sorted(station.bikes.values(), key=lambda bike: bike.battery, reverse=False)

                # Calculate hours until violation because no bikes have sufficient battery
                battery_top3 = [Ebike.battery for Ebike in sorted_bikes_in_station[-3:]]
                average_battery_top3 = sum(battery_top3)/len(battery_top3) if battery_top3 != [] else 0
                hourly_discharge = calculate_hourly_discharge_rate(simul, total_num_bikes_in_system)
                hours_until_violation_battery = average_battery_top3/hourly_discharge

                # Find the earlist moment for a violation
                hours_until_first_violation = min(
                                                (len(station.get_available_bikes())/ -net_demand), # How long until the net demand results in a starvation
                                                hours_until_violation_battery
                                                )
                
                # Find the time in minutes for the violation
                time_of_first_violation_no_visit = current_time + (hours_until_first_violation * 60)
                
            elif net_demand > 0:
                # How long until the net demand results in a congestion
                hours_until_first_violation = (station.capacity - station.number_of_bikes()) / net_demand
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
                    time_first_violation_after_visit = eta + min(time_until_first_violation, 100/calculate_hourly_discharge_rate(simul, total_num_bikes_in_system) * 60)
                else:
                    time_first_violation_after_visit = eta + min(time_until_first_violation, (average_battery_top3)/(calculate_hourly_discharge_rate(simul, total_num_bikes_in_system)) * 60)
            elif net_demand > 0:
                time_until_first_violation = (station.capacity - station_inventory_after_visit) / net_demand
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
                ending_inventory_after_visit = min(station.capacity, station_inventory_after_visit + ((end_time-eta)/60)*net_demand)
                ending_inventory_no_visit = min(station.capacity , initial_inventory + ((end_time-current_time)/60)*net_demand)

            deviation_after_visit = abs(ending_inventory_after_visit - target_state)
            deviation_no_visit = abs(ending_inventory_no_visit - target_state)

            improved_deviation = deviation_no_visit - deviation_after_visit

            # Calculate excess bikes and locks, with and wihtout visits
            excess_bikes_after_visit = ending_inventory_after_visit
            excess_locks_after_visit = station.capacity - ending_inventory_after_visit
            if net_demand > 0:
                excess_bikes_no_visit = min(station.capacity, initial_inventory + ((end_time-current_time)/60) * net_demand)
                excess_locks_no_visit = max(0, station.capacity - (initial_inventory + ((end_time-current_time)/60) * net_demand))
            elif net_demand <= 0:
                excess_bikes_no_visit = max(0, initial_inventory + ((end_time-current_time)/60) * net_demand)
                excess_locks_no_visit = min(station.capacity, station.capacity - (initial_inventory+((end_time-current_time)/60) * net_demand))
            
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

                distance_scaling = ((simul.state.get_vehicle_travel_time(station.location_id, neighbor.location_id)/60)*VEHICLE_SPEED)/MAX_ROAMING_DISTANCE_SOLUTIONS
                neighbor_roamings += (1 - distance_scaling) * (roamings - roamings_no_visit)
            
            avoided_disutility += discounting_factors[counter]*(weights[0]*avoided_violations + weights[1]*neighbor_roamings + weights[2]*improved_deviation)

            counter += 1
        
        return avoided_disutility

    def return_best_move_average(self, vehicle, simul, plan_scores):
        """
        Returns the ID of the Station with performing best on average over all the scenarios.

        Parameters:
        - vehicle = The Vehicle-object doing the action
        - simul = Simulator
        - plan_scores = dictionaries, key: Plan, value: list of float-scores for each scenario
        """

        if self.number_of_scenarios == 0:
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
            # simul.metrics.add_aggregate_metric(simul, "branch"+str(branch+1), 1)
            if branch is None:
                print(best_plan, "simul_time:", simul.time)
            first_move = best_plan.plan[vehicle.vehicle_id][1].station.location_id
            return first_move
        
        # If there is no best, choose a random station that is not in the tabu_list
        else: 
            tabu_list = [vehicle2.location.location_id for vehicle2 in simul.state.get_sb_vehicles()]
            potential_stations2 = [station for station in simul.state.get_stations() if station.location_id not in tabu_list]    
            rng_balanced = np.random.default_rng(None)
            return rng_balanced.choice(potential_stations2).location_id

def calculate_loading_quantities_and_swaps_greedy(vehicle, simul, station, congestion_criteria, starvation_criteria):
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
    target_state = round(station.get_target_state(simul.day(), simul.hour()))
    num_max_usable_bikes_after_visit = get_max_num_usable_bikes(station, vehicle)

    # Count how many neighbors are starved or congested
    starved_neighbors = 0
    congested_neighbors = 0
    for neighbor in station.neighbours:
        num_bikes_neighbor = len(neighbor.get_available_bikes())
        neighbor_target_state = round(neighbor.get_target_state(simul.day(), simul.hour()))
        if num_bikes_neighbor < starvation_criteria * neighbor_target_state:
            starved_neighbors += 1
        elif num_bikes_neighbor > congestion_criteria * neighbor_target_state:
            congested_neighbors += 1

    # If the station is a delivery station, calculate which bikes to deliver from the vehicle, and which bikes at the station to swap on
    if num_max_usable_bikes_after_visit < target_state:
        num_bikes_to_deliver = min(
            len([bike for bike in vehicle.get_sb_bike_inventory() if bike.battery > BATTERY_LIMIT_TO_USE]), 
            target_state - num_max_usable_bikes_after_visit + BIKES_STARVED_NEIGHBOR * starved_neighbors)
        bikes_to_deliver, bikes_to_swap = get_bike_ids_load_swap(station, vehicle, num_bikes_to_deliver, "deliver")
        bikes_to_pick_up = []
    
    # If the station is a pickup station, calculate which bikes to pickup, and which to bikes at the station to swap batteries on
    elif num_max_usable_bikes_after_visit > target_state:
        remaining_cap_vehicle = vehicle.bike_inventory_capacity - len(vehicle.get_sb_bike_inventory())
        num_bikes_to_pickup = min(
            remaining_cap_vehicle, 
            num_max_usable_bikes_after_visit - target_state + BIKES_OVERFLOW_NEIGHBOR * congested_neighbors, 
            len(station.bikes))
        bikes_to_deliver=[]
        bikes_to_pick_up, bikes_to_swap = get_bike_ids_load_swap(station, vehicle, num_bikes_to_pickup, "pickup")

    # If no bikes need to be picked up or delivered, find out how many bikes to swap batteries on
    else:
        bikes_to_pick_up = []
        bikes_to_deliver = []

        unusable_bikes_at_station = station.get_swappable_bikes(BATTERY_LIMIT_TO_USE)
        num_bikes_to_swap = min(
            len(unusable_bikes_at_station),
            vehicle.battery_inventory)
        bikes_to_swap = [escooter.bike_id for escooter in unusable_bikes_at_station[:num_bikes_to_swap]]

    # Return lists of bike IDs to do each action on
    return bikes_to_pick_up, bikes_to_deliver, bikes_to_swap

def get_max_num_usable_bikes(station, vehicle): 
    """"
    Returns max number of bikes at station with sufficient battery level, neglects bikes that cannot be fixed.

    Parameters:
    - station = Station being considered
    - vehicle = Vehicle considered to rebalance station
    """
    return len(station.get_available_bikes()) + min(len(station.get_swappable_bikes()), vehicle.battery_inventory)

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
        vehicle_bikes =  sorted(vehicle.get_sb_bike_inventory(), key=lambda bike: bike.battery, reverse=False)
    else:
        bikes_at_station = list(station.bikes.values())
        vehicle_bikes =  vehicle.get_sb_bike_inventory()

    # Returns lists of bike IDs on which to deliver and which to swap batteries on
    if station_type == "deliver":
        num_bikes_to_swap = min(len(station.get_swappable_bikes(BATTERY_LIMIT_TO_USE)), vehicle.battery_inventory)
        num_bikes_to_deliver = int(num_bikes)+1

        bikes_to_swap = [bike.bike_id for bike in bikes_at_station[:num_bikes_to_swap]]
        bikes_to_deliver = [bike.bike_id for bike in vehicle_bikes[-num_bikes_to_deliver:]]

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
        num_bikes_to_only_swap = min(vehicle.battery_inventory, max(0,len(station.get_swappable_bikes(BATTERY_LIMIT_TO_USE)) - num_bikes_to_swap_and_pickup))

        bikes_to_swap = []
        bikes_to_pickup = [bike.bike_id for bike in bikes_at_station[:num_bikes_to_swap_and_pickup]]
        if num_bikes_to_only_pickup > 0:
            bikes_to_pickup += [bike.bike_id for bike in bikes_at_station[-num_bikes_to_only_pickup:]]
        
        elif num_bikes_to_only_swap > 0:
            bikes_to_swap += [bike.bike_id for bike in bikes_at_station[num_bikes_to_swap_and_pickup:num_bikes_to_swap_and_pickup+num_bikes_to_only_swap]]

        return bikes_to_pickup, bikes_to_swap
    
    return [],[]

def find_potential_stations(simul, cutoff_vehicle, cutoff_station, vehicle, bikes_in_vehicle, tabu_list):
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
                                 if get_max_num_usable_bikes(station, vehicle) + net_demands[station.location_id] > (1 + cutoff_station) * target_states[station.location_id]
                                 ] # TODO blir cuttoff annerledes med station based og capacity?
    
    # If the available bikes after a visit in the future is lower that a cutoff percentage of target state, the station is a delivery station
    potential_delivery_stations = [ station for station in potential_stations
                                 if get_max_num_usable_bikes(station, vehicle) + net_demands[station.location_id] < (1 - cutoff_station) * target_states[station.location_id]
                                 ]
    
    #Decides pickup, delivery or balanced is relevant
    if cutoff_vehicle * vehicle.bike_inventory_capacity <= bikes_in_vehicle <= (1 - cutoff_vehicle) * vehicle.bike_inventory_capacity:
        potential_stations = potential_pickup_stations + potential_delivery_stations

    else:
        if bikes_in_vehicle <= cutoff_vehicle * vehicle.bike_inventory_capacity:
            potential_stations = potential_pickup_stations

        elif bikes_in_vehicle >= (1 - cutoff_vehicle) * vehicle.bike_inventory_capacity:
            potential_stations = potential_delivery_stations
    
    return potential_stations