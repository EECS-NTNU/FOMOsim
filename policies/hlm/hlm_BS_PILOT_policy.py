from policies import Policy
from settings import MAX_ROAMING_DISTANCE_SOLUTIONS, VEHICLE_SPEED, MINUTES_CONSTANT_PER_ACTION
import sim
from .Visit import Visit
from .Plan import Plan
from .Criticality_score import calculate_criticality, calculate_station_type
from .Simple_calculations import calculate_net_demand, copy_arr_iter, generate_discounting_factors, calculate_hourly_discharge_rate

import numpy as np
import time

########################################## 
#         BS_PILOT - policy class        #
##########################################

class BS_PILOT(Policy): #Add default values from seperate setting sheme
    def __init__(self, 
                 max_depth = 2, 
                 number_of_successors = 5, 
                 time_horizon = 40, 
                 criticality_weights_sets = [[0.3, 0.15, 0.25, 0.2, 0.1], [0.3, 0.5, 0, 0, 0.2], [0.6, 0.1, 0.05, 0.2, 0.05]], 
                 evaluation_weights = [0.85, 0.1, 0.05], 
                 number_of_scenarios = 100, 
                 discounting_factor = 0.1
                 ):
        self.max_depth = max_depth
        self.number_of_successors = number_of_successors
        self.time_horizon = time_horizon
        self.criticality_weights_set = criticality_weights_sets
        self.evaluation_weights = evaluation_weights
        self.number_of_scenarios = number_of_scenarios
        self.discounting_factor = discounting_factor
        super().__init__()

    ###############################################
    #  Returns an action element that contains:   #
    #  * Number of batteries to swap              #
    #  * Number of e-scooters to pick up          #
    #  * Number of e-scooters to deliver          #
    #  * Next station to visit                    #
    #                                             #
    # This is how our code talks to the simulator #            
    ###############################################
    
    def get_best_action(self, simul, vehicle):
        start_logging_time = time.time() 
        next_station = None
        escooters_to_pickup = []
        escooters_to_deliver = []
        batteries_to_swap = []

        end_time = simul.time + self.time_horizon 
        total_num_bikes_in_system = sum([station.number_of_bikes() for station in simul.state.stations.values()]) + len(simul.state.bikes_in_use) #flytt hvis lang kjøretid

        #### TODO NEW FUNCTIONALITY ####
        # depot = simul.state.depot

        # if vehicle.battery_inventory <= 0:
        #     next_station = self.find_closest_depot(simul, vehicle, depot)

        #     return sim.Action(
        #     [],
        #     [],
        #     [],
        #     next_station
        # )

        #########################################################################################
        #   Number of bikes to pick up / deliver is choosen greedy based on clusters in reach   #
        #   Which bike ID´s to pick up / deliver is choosen based on battery level              #   
        #   How many batteries to swap choosen based on battery inventory and status on station #           
        #########################################################################################

        escooters_to_pickup, escooters_to_deliver, batteries_to_swap = calculate_loading_quantities_and_swaps_greedy(vehicle, simul, vehicle.location)
        number_of_escooters_pickup = len(escooters_to_pickup)
        number_of_escooters_deliver = len(escooters_to_deliver)
        number_of_batteries_to_swap = len(batteries_to_swap)


        ######################################################################################################################
        #  If estimated time of arrival (eta) is zero, meaning that the vehicle is at a station keep greedy quantity values  #                     #
        #  If eta is in the future (max 60 minutes ahead) find new quantities with future demand taken into consideration    #
        ######################################################################################################################

        plan_dict = dict()
        for v in simul.state.vehicles:
            if v.eta == 0:
                plan_dict[v.id] = [Visit(v.location, number_of_escooters_pickup, number_of_escooters_deliver, number_of_batteries_to_swap, simul.time, v)]
            else:
                number_of_escooters_pickup, number_of_escooters_deliver, number_of_batteries_to_swap = self.calculate_loading_quantities_and_swaps_pilot(v, simul, v.location, v.eta) #I think eta is estimated time of arrival
                plan_dict[v.id] = [Visit(v.location, int(number_of_escooters_pickup), int(number_of_escooters_deliver), int(number_of_batteries_to_swap), v.eta, v)]
        
        tabu_list = [v.location.id for v in simul.state.vehicles]
        plan = Plan(plan_dict, tabu_list)


        ###############################################
        #  Use PILOT_function to decide next station  #
        ###############################################

        next_station = self.PILOT_function(simul, vehicle, plan, self.max_depth, self.number_of_successors, end_time, total_num_bikes_in_system)

        similary_imbalances_starved = 0
        similary_imbalances_overflow = 0

        for neighbor in simul.state.stations[next_station].neighboring_stations:
            if neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(20)) > 1.1 * neighbor.get_target_state(simul.day(),simul.hour()) and number_of_escooters_pickup > 0:
                similary_imbalances_overflow += 1
            elif neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(20)) < 0.1 * neighbor.get_target_state(simul.day(),simul.hour()) and number_of_escooters_deliver > 0:
                similary_imbalances_starved += 1
            
        
        simul.metrics.add_aggregate_metric(simul, "similarly imbalanced starved", similary_imbalances_starved)
        simul.metrics.add_aggregate_metric(simul, "similarly imbalanced congested", similary_imbalances_overflow)
        simul.metrics.add_aggregate_metric(simul, "accumulated solution time", time.time()-start_logging_time)
        simul.metrics.add_aggregate_metric(simul, 'number of problems solved', 1)

        return sim.Action(
            batteries_to_swap,
            escooters_to_pickup,
            escooters_to_deliver,
            next_station
        )



    #################################################################################################### 
    # PILOT_function - optimizes next station based on many possible future outcomes from that choice  #
    ####################################################################################################

    def PILOT_function(self, simul, vehicle, initial_plan, max_depth, number_of_successors, end_time, total_num_bikes_in_system):
        completed_plans = []
        for weight_set in self.criticality_weights_set:
            num_successors = number_of_successors
            plans = [[] for i in range(max_depth +1)] # [[], [], [], ... ] -> len() = max_depth+1
            plans[0].append(initial_plan) # plan of visits so far
            depths = [i for i in range (1, max_depth+1)] # [1, 2, 3, ... ] -> len() = max_depth

            for depth in depths:
                if depth == 2 or depth == 3:
                    num_successors = max(1, round(num_successors/2)) # halve beta after depth 1 and 2
                
                while plans[depth-1] != []:
                    plan = plans[depth-1].pop(0)
                    next_vehicle = plan.next_visit.vehicle
                    if next_vehicle != vehicle:
                        num_successors_other_veheicle = max(1, round(num_successors/2))
                        new_visits = self.greedy_next_visit(plan, simul, num_successors_other_veheicle, weight_set, total_num_bikes_in_system)
                    else:
                        new_visits = self.greedy_next_visit(plan, simul, num_successors, weight_set, total_num_bikes_in_system)
                    
                    if new_visits == None or plan.next_visit.get_depature_time() > end_time:
                        new_plan = Plan(plan.copy_plan(),copy_arr_iter(plan.tabu_list), weight_set, plan.branch_number)
                        plans[depth].append(new_plan)
                    else:
                        for branch_number, visit in enumerate(new_visits):
                            new_plan_dict = plan.copy_plan()
                            new_plan_dict[next_vehicle.id].append(visit)
                            tabu_list = copy_arr_iter(plan.tabu_list)
                            tabu_list.append(visit.station.id)

                            if depth == 1:
                                new_plan = Plan(new_plan_dict, tabu_list, weight_set, branch_number)
                            else:
                                new_plan = Plan(new_plan_dict, tabu_list, weight_set, plan.branch_number)
                            

                            if next_vehicle.id == vehicle.id:
                                plans[depth].append(new_plan)
                            else:
                                plans[depth-1].append(new_plan)


            for plan in plans[max_depth]:

                dep_time = plan.next_visit.get_depature_time()
                temp_plan = Plan(plan.copy_plan(), copy_arr_iter(plan.tabu_list), weight_set, plan.branch_number)

                while dep_time < end_time:
                    new_visit = self.greedy_next_visit(temp_plan, simul, 1, weight_set, total_num_bikes_in_system)
                    
                    if new_visit != None:
                        new_visit = new_visit[0]
                        temp_plan.tabu_list.append(new_visit.station.id)
                    else:
                        break

                    
                    temp_plan.plan[temp_plan.next_visit.vehicle.id].append(new_visit)
                    dep_time = new_visit.get_depature_time()
                    temp_plan.find_next_visit()
                
                completed_plans.append(temp_plan)
            
        plan_scores = dict()

        #Generates different demand scenarios?
        scenarios = self.generate_scenarioes(simul, self.number_of_scenarios, poisson = True)

        for plan in completed_plans:
            plan_scores[plan] = []
            for scenario_dict in scenarios:
                score = 0
                for v in plan.plan:
                    score += self.evaluate_route(plan.plan[v], scenario_dict, end_time, simul, self.evaluation_weights, total_num_bikes_in_system)
                plan_scores[plan].append(score)
        
        # Returns the station with the best avarage score over all scenarios
        return self.return_best_move_avarage(vehicle, simul, plan_scores)

                
                    


    

    ################################################################################################
    # An extention of the greedy alorithm, only difference is that future demand is accounted for  #
    # Thus this does only return quantities, not id´s                                              # 
    # Demand is calculated for the next hour, and treathed as evenly distrubuted trough that hour  #
    ################################################################################################
    def calculate_loading_quantities_and_swaps_pilot(self, vehicle, simul, station, eta):
        num_escooters_vehicle = len(vehicle.get_bike_inventory())

        number_of_escooters_pickup = 0
        number_of_escooters_deliver = 0
        number_of_escooters_swap = 0


        target_state = round(station.get_target_state(simul.day(),simul.hour())) 
        net_demand = calculate_net_demand(station, simul.time, simul.day(), simul.hour(), 60) # Gives net_demand per hour
        num_escooters_accounted_for_battery_swaps = get_num_escooters_accounted_for_battery_swaps(station, station.number_of_bikes(), vehicle)
        num_escooters_station_at_arrival_accounted_battery_swap = num_escooters_accounted_for_battery_swaps + ((eta - simul.time)/60)*net_demand # How many bikes at station at eta, based on demand forecast

        starved_neighbors = 0
        overflowing_neighbors = 0

        for neighbor in station.neighboring_stations:
            net_demand_neighbor = calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(),60)
            num_escooters_neighbor = neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(20)) + ((eta - simul.time)/60)*net_demand_neighbor
            neighbor_target_state = round(neighbor.get_target_state(simul.day(), simul.hour()))
            if num_escooters_neighbor < 0.1 * neighbor_target_state:
             starved_neighbors += 1
            elif num_escooters_neighbor > 1.1 * neighbor_target_state:
             overflowing_neighbors += 1

        if num_escooters_station_at_arrival_accounted_battery_swap < target_state:
            number_of_additional_escooters = min(num_escooters_vehicle, target_state - num_escooters_station_at_arrival_accounted_battery_swap + 2*starved_neighbors)
            escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = id_escooters_accounted_for_battery_swaps(station, vehicle, number_of_additional_escooters, "deliver")
            number_of_escooters_deliver = len(escooters_to_deliver_accounted_for_battery_swaps)
            number_of_escooters_swap = len(escooters_to_swap_accounted_for_battery_swap)
        
        elif num_escooters_station_at_arrival_accounted_battery_swap > target_state:
            remaining_cap_vehicle = vehicle.bike_inventory_capacity - num_escooters_vehicle
            number_of_less_escooters = min(remaining_cap_vehicle, num_escooters_station_at_arrival_accounted_battery_swap - target_state + 2*overflowing_neighbors)
            escooters_to_pickup_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = id_escooters_accounted_for_battery_swaps(station, vehicle, number_of_less_escooters, "pickup")
            number_of_escooters_pickup = len(escooters_to_pickup_accounted_for_battery_swaps)
            number_of_escooters_swap = len(escooters_to_swap_accounted_for_battery_swap)

        else:
            escooters_in_station_low_battery = station.get_swappable_bikes(20)
            number_of_escooters_swap = min(len(escooters_in_station_low_battery), vehicle.battery_inventory)
        
        return number_of_escooters_pickup, number_of_escooters_deliver, number_of_escooters_swap
    


    ###################################################################
    # Finds next station to visit greedy based on criticallity scores #
    ###################################################################
    def greedy_next_visit(self, plan, simul, number_of_successors, weight_set, total_num_bikes_in_system):
        visits = []
        tabu_list = plan.tabu_list
        vehicle = plan.next_visit.vehicle

        initial_num_escooters = len(vehicle.get_bike_inventory())
        num_bikes_now = initial_num_escooters

        for visit in plan.plan[vehicle.id]: #la til index -1 (gir dette mening?)
            num_bikes_now += visit.loading_quantity
            num_bikes_now -= visit.unloading_quantity

        # Finds potential next stations based on pick up or delivery status of the station and tabulist
        potential_stations, station_type = find_potential_stations(simul,0.15,0.15,vehicle, num_bikes_now, tabu_list) #Hvor kommer 0.15 fra??
        if potential_stations == []: #Try to find out when and if this happens?
            return None
        
        number_of_successors = min(number_of_successors, len(potential_stations))

        # Finds the criticality score of all potential stations
        stations_sorted = calculate_criticality(weight_set, simul, potential_stations, plan.plan[vehicle.id][-1].station,station_type, total_num_bikes_in_system ,tabu_list)
        stations_sorted_list = list(stations_sorted.keys())
        next_stations = [stations_sorted_list[i] for i in range(number_of_successors)]

        for next_station in next_stations:
            arrival_time = plan.plan[vehicle.id][-1].get_depature_time() + simul.state.traveltime_vehicle_matrix[plan.plan[vehicle.id][-1].station.id][next_station.id] + MINUTES_CONSTANT_PER_ACTION
            number_of_escooters_to_pickup, number_of_escooters_to_deliver, number_of_escooters_to_swap = self.calculate_loading_quantities_and_swaps_pilot(vehicle, simul, next_station, arrival_time)
            new_visit = Visit(next_station, number_of_escooters_to_pickup, number_of_escooters_to_deliver, number_of_escooters_to_swap, arrival_time, vehicle)
            visits.append(new_visit)
        
        return visits
    

    ####################################################################
    # I dont really understand this function - have copied it as i was #
    # If any of you understands - hit me up                            #
    ####################################################################

    def generate_scenarioes(self, simul, number_of_scenarios, poisson = True): #normal_dist if poisson = False
        rng = np.random.default_rng(simul.state.seed) 
        scenarios = []
        stations_dict = simul.state.stations 
        if number_of_scenarios < 1: #0, return expected net_demand values
            scenario_dict = dict() #station_id : net demand
            for station_id in stations_dict:
                net_demand =  calculate_net_demand(stations_dict[station_id], simul.time ,simul.day(),simul.hour(), 60) #returns net demand for next hour 
                scenario_dict[station_id] = net_demand
            scenarios.append(scenario_dict)
        
        else:
            for s in range(number_of_scenarios):
                scenario_dict = dict()
                planning_horizon = 60 #calculate net_demand for the next 60 minutes 
                time_now = simul.time
                day = simul.day()
                hour = simul.hour()
                minute_in_current_hour = time_now-day*24*60-hour*60 
                minutes_current_hour = min(60-minute_in_current_hour,planning_horizon)
                minutes_next_hour = planning_horizon - minutes_current_hour
                
                for station_id in stations_dict: 
                    expected_arrive_intensity = 2*stations_dict[station_id].get_arrive_intensity(simul.day(), simul.hour())
                    expected_leave_intensity = 2*stations_dict[station_id].get_leave_intensity(simul.day(), simul.hour())
                    expected_arrive_intensity_next = 2*stations_dict[station_id].get_arrive_intensity(simul.day(), simul.hour()+1)
                    expected_leave_intensity_next = 2*stations_dict[station_id].get_leave_intensity(simul.day(), simul.hour()+1)
                    
                    if poisson:
                        net_demand_current = rng.poisson(expected_arrive_intensity) - rng.poisson(expected_leave_intensity)
                        net_demand_next = rng.poisson(expected_arrive_intensity_next) - rng.poisson(expected_leave_intensity_next)
                        net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/planning_horizon
                    
                    else: #normal_dist
                        arrive_intensity_stdev = stations_dict[station_id].get_arrive_intensity_stdev(simul.day(), simul.hour())
                        leave_intensity_stdev = stations_dict[station_id].get_leave_intensity_stdev(simul.day(), simul.hour())
                        arrive_intensity_stdev_next = stations_dict[station_id].get_arrive_intensity_stdev(simul.day(), simul.hour()+1)
                        leave_intensity_stdev_next = stations_dict[station_id].get_leave_intensity_stdev(simul.day(), simul.hour()+1)

                        net_demand_current = rng.normal(expected_arrive_intensity, arrive_intensity_stdev) - rng.normal(expected_leave_intensity, leave_intensity_stdev)
                        net_demand_next = rng.normal(expected_arrive_intensity_next, arrive_intensity_stdev_next) - rng.normal(expected_leave_intensity_next, leave_intensity_stdev_next)
                        net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/planning_horizon
                        
                    scenario_dict[station_id] = net_demand 
                scenarios.append(scenario_dict)
        return scenarios
    


    #####################################################################################
    # Evaluating how much extra utility we get by choosing a route                      #
    # Weights for how much to value avoided violations, improved diviation and roamings #
    # Discounting factors discounts over time (i think)                                 #
    #####################################################################################

    def evaluate_route(self, route, scenario_dict, end_time, simul, weights, total_num_bikes_in_system):
        # Begins with current station and loading quantities

        discounting_factors = generate_discounting_factors(len(route), self.discounting_factor)
        avoided_disutility = 0
        current_time = simul.time
        counter = 0

        for visit in route:
            avoided_violations = 0
            neighbor_roamings = 0
            improved_deviation = 0

            station = visit.station

            #pick up quantity
            loading_quantity = visit.loading_quantity
            #deliver quantity
            unloading_quantity = visit.unloading_quantity
            #swap quantity
            swap_quantity = visit.swap_quantity

            neighbors = station.neighboring_stations

            eta = visit.arrival_time

            if eta > end_time:
                eta = end_time
            
            initial_inventory = station.number_of_bikes() - len(station.get_swappable_bikes(20))
            net_demand = scenario_dict[station.id]
            target_state = station.get_target_state(simul.day(), simul.hour())

            #########################################################################
            # AVOIDED VIOLATIONS                                                    #
            # Below we implement all the ways we want to measure avoided violations #
            # I have removed conjuctions as a violation                             #
            #########################################################################

            if net_demand < 0:
                sorted_escooters_in_station = sorted(station.bikes.values(), key=lambda bike: bike.battery, reverse=False)
                time_first_violation_no_visit = current_time + min((station.number_of_bikes() - len(station.get_swappable_bikes(20)))/ -net_demand, (sum(Ebike.battery for Ebike in sorted_escooters_in_station[-3:])/3)/(calculate_hourly_discharge_rate(simul, total_num_bikes_in_system)*60))
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
                    time_first_violation_after_visit = eta + min((inventory_after_loading_and_swaps/(-net_demand))*60, (sum(Ebike.battery for Ebike in sorted_escooters_in_station[-3:])/3)/(calculate_hourly_discharge_rate(simul, total_num_bikes_in_system)*60))
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
                net_demand_neighbor = scenario_dict[neighbor.id]
                expected_ecooters_neighbor = neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(20)) + net_demand_neighbor
                neighbor_type = calculate_station_type(neighbor.get_target_state(simul.day(),simul.hour()),expected_ecooters_neighbor)

                if neighbor_type == station_type:
                    if net_demand_neighbor < 0:
                        time_first_violation = current_time + ((neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(20)))/-net_demand_neighbor) * 60
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
                        
            
                distance_scaling = ((simul.state.get_vehicle_travel_time(station.id, neighbor.id)/60)* VEHICLE_SPEED)/MAX_ROAMING_DISTANCE_SOLUTIONS
                neighbor_roamings += (1-distance_scaling)*roamings-roamings_no_visit
            
            avoided_disutility += discounting_factors[counter]*(weights[0]*avoided_violations + weights[1]*neighbor_roamings + weights[2]*improved_deviation)

            counter += 1
        
        return avoided_disutility
    

    ##########################################################################
    # Finds the action which on avarage performs best over several scenarios #
    ##########################################################################

    def return_best_move_avarage(self, vehicle, simul, plan_scores):
        score_board = dict() #plan object: the average score of this plan
        num_scenarios=self.number_of_scenarios
        if num_scenarios==0:
            num_scenarios+=1 #this scenario is now the expected value 
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
            first_move = best_plan.plan[vehicle.id][1].station.id
            return first_move
        else: 
            tabu_list = [vehicle2.location.id for vehicle2 in simul.state.vehicles]
            potential_stations2 = [station for station in simul.state.locations if station.id not in tabu_list]    
            rng_balanced = np.random.default_rng(None)
            return rng_balanced.choice(potential_stations2).id

    ####################################################################
    # Finds closest depot from location when vehicle is out of battery #
    ####################################################################

    def find_closest_depot(self, simul, vehicle, depot):
        closest_depot = None
        closest_distance = 100000000


        for d in depot:
            distance = (simul.state.traveltime_vehicle_matrix[vehicle.location.id][d.id]/60)*VEHICLE_SPEED
            if distance < closest_distance:
                closest_distance = distance
                closest_depot = d

        return closest_depot
                        


#############################################################################################
#   Number of bikes to pick up / deliver is choosen greedy based on clusters in reach       #
#   Which bike ID´s to pick up / deliver is choosen based on battery level                  #
#   How many and which escooters to swap battery on based on inventory and station status   #                                      
#   Applied functionality such that scooters with battery level < thershold does not count  #             
#############################################################################################

def calculate_loading_quantities_and_swaps_greedy(vehicle, simul, station):
    num_escooters_vehicle = len(vehicle.get_bike_inventory())

    target_state = round(station.get_target_state(simul.day(), simul.hour())) #Denne må vi finne ut hvordan lages
    num_escooters_station = station.number_of_bikes() # number of scooters at the station
    num_escooters_accounted_for_battery_swaps = get_num_escooters_accounted_for_battery_swaps(station, num_escooters_station, vehicle) # 

    ################################################################
    #  Adjusting numbers based on status at neighboring stations   #
    #  Changed from capacity to diviation from ideal state         #
    ################################################################

    starved_neighbors = 0
    overflowing_neighbors = 0

    for neighbor in station.neighboring_stations:
        num_escooters_neighbor = neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(20)) #Bytt ut med et dokument tilsvarende settings
        neighbor_target_state = round(neighbor.get_target_state(simul.day(), simul.hour()))
        if num_escooters_neighbor < 0.1 * neighbor_target_state:
            starved_neighbors += 1
        elif num_escooters_neighbor > 1.1 * neighbor_target_state:
            overflowing_neighbors += 1

    #######################################################################
    #  Find out whether the station is a pick-up or a deliver station     #
    #  And based on that the quantities for pickup, deliveries and swaps  #
    #######################################################################

    if num_escooters_accounted_for_battery_swaps < target_state: #Ta hensyn til nabocluster her?
        number_of_escooters_to_deliver = min(len([escooter for escooter in vehicle.get_bike_inventory() if escooter.battery > 20]), target_state - num_escooters_accounted_for_battery_swaps + 2*starved_neighbors) # discuss 2*starved_neighbors part, ta hensyn til postensielle utladede scootere i bilen
        escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = id_escooters_accounted_for_battery_swaps(station, vehicle, number_of_escooters_to_deliver, "deliver")
        escooters_to_pickup_accounted_for_battery_swaps = []
    
    elif num_escooters_accounted_for_battery_swaps > target_state:
        remaining_cap_vehicle = vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory())
        number_of_escooters_to_pickup = min(remaining_cap_vehicle, num_escooters_accounted_for_battery_swaps - target_state + 2*overflowing_neighbors, len(station.bikes)) #discuss logic behind this
        escooters_to_deliver_accounted_for_battery_swaps=[]
        escooters_to_pickup_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = id_escooters_accounted_for_battery_swaps(station, vehicle, number_of_escooters_to_pickup, "pickup")

    else:
        escooters_to_pickup_accounted_for_battery_swaps = []
        escooters_to_deliver_accounted_for_battery_swaps = []

        escooters_in_station_low_battery = station.get_swappable_bikes(20)
        num_escooters_to_swap = min(len(escooters_in_station_low_battery),vehicle.battery_inventory)
        escooters_to_swap_accounted_for_battery_swap = [escooter.id for escooter in escooters_in_station_low_battery[:num_escooters_to_swap]]


    
    return escooters_to_pickup_accounted_for_battery_swaps, escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap




#######################################################################################################################
# Simple calculation function to take low battery level into consideration when choosing number to deliver or pickup  #
#######################################################################################################################

def get_num_escooters_accounted_for_battery_swaps(station, num_escooters_station, vehicle): 
    """"
    Neglects the scooters what cannot be fixed

    Returns max number of scooters at station with sufficient battery level
    """
    return num_escooters_station - len(station.get_swappable_bikes(20)) + min(len(station.get_swappable_bikes(20)),vehicle.battery_inventory)


############################################################################################
# If deliver, swap as many low battries in station as possible then deliver rest of bikes  #
# Swaps from low battry until threshold, default 20                                        #
# Delivers from high battery until threshold                                               #  
#                                                                                          #
# If pickup, pickup and swap as many escooters with low battry as possible                 #
# Then pickup rest from high battery end or swap remaining swaps                           #
############################################################################################

def id_escooters_accounted_for_battery_swaps(station, vehicle, number_of_escooters, type):
    escooters_in_station = sorted(station.bikes.values(), key=lambda bike: bike.battery, reverse=False) # escooters at station, sorted from lowest to highest battery level. List consists of bike-objects
    escooters_in_vehicle =  sorted(vehicle.get_bike_inventory(), key=lambda bike: bike.battery, reverse=False) # escooters in vehicle, sortes from lowest to highest bettery level. List consists of bike-objects

    if type == "deliver":
        number_of_escooters_to_swap = min(len(station.get_swappable_bikes(20)),vehicle.battery_inventory) # get all the bikes under 20% if available capacity, otherwise swap batteries on n bikes with the lowest battery level. n = available capacity in vehicle
        #TODO Burde vi runde opp eller ned?
        number_of_escooters_to_deliver = int(number_of_escooters)+1 # If 4.2 escooter are being delivered, we deliver 5

        escooters_to_swap = [escooter.id for escooter in escooters_in_station[:number_of_escooters_to_swap]] # swap the n escooters with lowest percentage at the station
        escooters_to_deliver = [escooter.id for escooter in escooters_in_vehicle[-number_of_escooters_to_deliver:]] # unload the n escooters with the highest percentage

        return escooters_to_deliver, escooters_to_swap
    
    elif type == "pickup":
        #TODO Burde vi runde på en annen måte? Nå runder vi til nærmeste heltall
        number_of_escooters_to_swap_and_pickup = min(len(station.get_swappable_bikes(70)),vehicle.battery_inventory, round(number_of_escooters)) # decide threshold for swap, either decided by # of full batteries, # of escooters under 70% and the max amount we are allowed to pick up based om target
        number_of_escooters_to_only_pickup = round(number_of_escooters) - number_of_escooters_to_swap_and_pickup # how many to pick up
        number_of_escooters_to_only_swap = max(0,len(station.get_swappable_bikes(20)) - number_of_escooters_to_swap_and_pickup)

        escooters_to_swap = []
        escooters_to_pickup = [escooter.id for escooter in escooters_in_station[:number_of_escooters_to_swap_and_pickup]]
        if number_of_escooters_to_only_pickup > 0:
            escooters_to_pickup += [escooter.id for escooter in escooters_in_station[-number_of_escooters_to_only_pickup:]]
        
        elif number_of_escooters_to_only_swap > 0:
            escooters_to_swap += [escooter.id for escooter in escooters_in_station[number_of_escooters_to_swap_and_pickup:number_of_escooters_to_swap_and_pickup+number_of_escooters_to_only_swap]]

        return escooters_to_pickup, escooters_to_swap
    
    return [],[]


###############################################################################################################################
# Considers all stations exept those in tabu_list                                                                             # 
# Cutoff_vehicle - a value between 0-1 that helps decide wheater to do a pick up or a delivery                                # 
# Cutoff_station - a value between 0-1 that helps decide wheater a station is a pickup or a delivery station                  #
# Seems like it is dependent on vehicle inventory levels alone i think we could consider including distances here             #
###############################################################################################################################

def find_potential_stations(simul, cutoff_vehicle, cutoff_station, vehicle, bikes_at_vehicle, tabu_list):

    # Filter out stations in tabulist
    potential_stations = [station for station in simul.state.locations if station.id not in tabu_list]

    # Makes dictonary with all potential stations and their respective demands
    net_demands = {station.id : calculate_net_demand(station, simul.time, simul.day(), simul.hour(), 60) for station in potential_stations}

    # Makes dictionary with all potential stations and their respective target states
    target_states = {station.id : station.get_target_state(simul.day(), simul.hour()) for station in potential_stations}

    # Finds the stations from potential stations which would be a pickup station if choosen - accounted for battery level / swaps
    potential_pickup_stations = [station for station in potential_stations if get_num_escooters_accounted_for_battery_swaps(station, station.number_of_bikes(), vehicle) + net_demands[station.id] > (1 + cutoff_station)*target_states[station.id]]

    # Finds the stations from potential stations which woukd be a delivery station if choosen - accounted for battery level / swaps
    potential_delivery_stations = [station for station in potential_stations if get_num_escooters_accounted_for_battery_swaps(station, station.number_of_bikes(), vehicle) + net_demands[station.id] < (1 - cutoff_station)*target_states[station.id]]

    type = 'b'
    
    #Decides pickup, delivery or both is relevant for 
    if cutoff_vehicle * vehicle.bike_inventory_capacity <= bikes_at_vehicle <= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
        potential_stations = potential_pickup_stations + potential_delivery_stations

    else:
        if bikes_at_vehicle <= cutoff_vehicle*vehicle.bike_inventory_capacity:
            potential_stations = potential_pickup_stations
            type = 'p'

        elif bikes_at_vehicle >= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
            potential_stations = potential_delivery_stations
            type = 'd'
    
    return potential_stations, type
