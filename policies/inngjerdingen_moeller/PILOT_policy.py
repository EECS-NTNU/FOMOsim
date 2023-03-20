'''rename this module and class eg., PILOT, InngjerdingenMoellerPolicy_2, heuristic or something similar '''
from policies import Policy
import sim
from criticality_score_neighbor import calculate_criticality, calculate_station_type
import settings
from policies.gleditsch_hagen.utils import calculate_net_demand
from greedy_policy_with_neighbors import calculate_loading_quantities_greedy
from greedy_policy_with_neighbors import find_potential_stations
import numpy as np

class PILOT(Policy):
    def __init__(self, max_depth=3, number_of_successors=3, time_horizon=60, criticality_weights_sets=[[0.2, 0.2, 0.2, 0.2, 0.2]], evaluation_weights=[0.33, 0.33, 0.33], number_of_scenarios=1):
        self.max_depth = max_depth
        self.number_of_successors = number_of_successors
        self.time_horizon = time_horizon
        self.crit_weights_sets = criticality_weights_sets
        self.evaluation_weights = evaluation_weights
        self.number_of_scenarios = number_of_scenarios
        super().__init__()

    def get_best_action(self, simul, vehicle):
        next_station = None 
        bikes_to_pickup = []
        bikes_to_deliver = []  
        
        end_time = simul.time + self.time_horizon 

        #########################################
        #               WHAT TO DO              #
        #########################################
        bikes_to_pickup, bikes_to_deliver = calculate_loading_quantities_greedy(vehicle, simul, vehicle.location)
        number_of_bikes_to_pick_up = len(bikes_to_pickup)
        number_of_bikes_to_deliver = len(bikes_to_deliver)
        
        
        plan_dict = dict()
        for v in simul.state.vehicles:
            if v.eta == 0:
                plan_dict[v.id] = [Visit(v.location, number_of_bikes_to_pick_up, number_of_bikes_to_deliver, simul.time, v)]
            else:
                number_of_bikes_to_pick_up, number_of_bikes_to_deliver = self.calculate_loading_quantities_pilot(v, len(v.get_bike_inventory()), simul, v.location, v.eta)
                plan_dict[v.id] = [Visit(v.location, int(number_of_bikes_to_pick_up), int(number_of_bikes_to_deliver), v.eta, v)]
        
        tabu_list = [v.location.id for v in simul.state.vehicles]
        
        plan = Plan(plan_dict, tabu_list)
    
        ##########################################
        #               WHERE TO GO              #
        ##########################################

        next_station = self.PILOT_function_multi_vehicle_multi_crit_weigths(simul, vehicle, plan, self.max_depth, self.number_of_successors, end_time)
        
        return sim.Action(
            [],               # batteries to swap
            bikes_to_pickup, #list of bike id's
            bikes_to_deliver, #list of bike id's
            next_station, #id 
        )   

    
    def PILOT_function(self, simul, vehicle, route, max_depth, number_of_successors, end_time): 
        routes = [[] for i in range(max_depth+1)]
        routes[0].append(route) 
        depth=0 
        min_departure_time = route[-1].get_departure_time()
        while depth < max_depth and min_departure_time < end_time:
            if depth == 1 or depth == 2:    # depth decreasing after first and second depth
                number_of_successors = max(1, round(number_of_successors/2))
            for route in routes[depth]:
                if route[-1].get_departure_time() < end_time and len(route)-1 < max_depth:
                    new_visits = self.greedy_next_visit(route, vehicle, simul, number_of_successors)
                    if new_visits == None:
                        new_route = copy_arr_iter(route)
                        routes[depth+1].append(new_route)
                        break
                    for visit in new_visits:
                        new_route = copy_arr_iter(route)
                        new_route.append(visit)
                        routes[depth+1].append(new_route) 
            depth+=1
            list_of_departure_times=[r[-1].get_departure_time() for r in routes[depth]]
            min_departure_time = min(list_of_departure_times)
           
        # Greedy construction for the rest of the route
        completed_routes = []
        for route in routes[depth]:
            dep_time = route[-1].get_departure_time()
            temp_route = copy_arr_iter(route)
            while dep_time < end_time:
                new_visit = self.greedy_next_visit(temp_route, vehicle, simul, 1)
                if new_visit != None:
                    new_visit = new_visit[0]
                else:
                    break
                temp_route.append(new_visit)
                dep_time = new_visit.get_departure_time()
            completed_routes.append(temp_route)

        route_scores = dict()
        for route in completed_routes:
            score = self.evaluate_route(route, None, end_time, simul, self.evaluation_weights)
            route_scores[tuple(route)]=score
        
        routes_sorted = dict(sorted(route_scores.items(), key=lambda item: item[1], reverse=True))
        best_route = list(routes_sorted.keys())[0]

        if len(best_route) < 2:     #no new stations to visit
            tabu_list = [vehicle2.location.id for vehicle2 in simul.state.vehicles]
            potential_stations2 = [station for station in simul.state.locations if station.id not in tabu_list]    
            rng_balanced = np.random.default_rng(None)
            return rng_balanced.choice(potential_stations2).id
        
        return best_route[1].station.id

    def PILOT_function_multi_vehicle(self, simul, vehicle, plan, max_depth, number_of_successors, end_time):     
        plans = [[] for i in range(max_depth+2)]
        plans[0].append(plan)
        depths = [i for i in range(max_depth+1)] 
        depth=0 
        for depth in depths:
            if depth == 1 or depth == 2:    # depth decreasing after first and second depth
                number_of_successors = max(1, round(number_of_successors/2))
            
            while plans[depth] != []:
                plan = plans[depth].pop(0)
                new_visits = self.greedy_next_visit(plan, simul, number_of_successors)
                next_vehicle = plan.next_visit.vehicle
                if new_visits == None or plan.next_visit.get_departure_time() > end_time:
                    new_plan = Plan(plan.copy_plan(), copy_arr_iter(plan.tabu_list))
                    plans[depth+1].append(new_plan)
                else:
                    for visit in new_visits:
                        new_plan_dict = plan.copy_plan()
                        new_plan_dict[next_vehicle.id].append(visit) 
                        tabu_list = copy_arr_iter(plan.tabu_list)
                        tabu_list.append(visit.station.id)
                        new_plan = Plan(new_plan_dict,tabu_list)

                        if next_vehicle.id == vehicle.id:
                            plans[depth+1].append(new_plan)
                        else:
                            plans[depth].append(new_plan)

                # if route[-1].get_departure_time() < end_time and len(route)-1 < max_depth
        
        # Greedy construction for the rest of the route
        completed_plans = []
        for plan in plans[max_depth+1]:
            dep_time = plan.next_visit.get_departure_time()
            temp_plan = Plan(plan.copy_plan(), copy_arr_iter(plan.tabu_list))
            while dep_time < end_time:
                new_visit = self.greedy_next_visit(temp_plan, simul, 1)
                if new_visit != None:
                    new_visit = new_visit[0]
                    temp_plan.tabu_list.append(new_visit.station.id)
                else:
                    break

                temp_plan.plan[temp_plan.next_visit.vehicle.id].append(new_visit)
                dep_time = new_visit.get_departure_time()
                temp_plan.find_next_visit()
            completed_plans.append(temp_plan)

        plan_scores = dict() #{plan_object: list of scenario_scores}

        scenarios = self.generate_scenarioes(simul, self.number_of_scenarios, self.time_horizon)

        for plan in completed_plans:
            plan_scores[plan] = []
            for scenario_dict in scenarios:
                score = 0
                for v in plan.plan:
                    score += self.evaluate_route(plan.plan[v], scenario_dict, end_time, simul, self.evaluation_weights)    
                plan_scores[plan].append(score)
        
        return self.return_best_move(vehicle, simul, plan_scores)
           
    def PILOT_function_multi_vehicle_multi_crit_weigths(self, simul, vehicle, initial_plan, max_depth, number_of_successors, end_time):     
        completed_plans = []
        for weight_set in self.crit_weights_sets:
            num_successors = number_of_successors
            plans = [[] for i in range(max_depth+2)]
            plans[0].append(initial_plan)
            depths = [i for i in range(max_depth+1)] 

            for depth in depths:
                if depth == 1 or depth == 2:    # depth decreasing after first and second depth
                    num_successors = max(1, round(num_successors/2))
                
                while plans[depth] != []:
                    plan = plans[depth].pop(0)
                    next_vehicle = plan.next_visit.vehicle
                    if next_vehicle != vehicle:
                        num_successors_other_vehicle = max(1, round(num_successors/2))
                        new_visits = self.greedy_next_visit(plan, simul, num_successors_other_vehicle, weight_set)
                    else:
                        new_visits = self.greedy_next_visit(plan, simul, num_successors, weight_set)
                    if new_visits == None or plan.next_visit.get_departure_time() > end_time:
                        new_plan = Plan(plan.copy_plan(), copy_arr_iter(plan.tabu_list), plan.weight_set, plan.branch_number)
                        plans[depth+1].append(new_plan)
                    else:
                        for branch_number, visit in enumerate(new_visits):
                            new_plan_dict = plan.copy_plan()
                            new_plan_dict[next_vehicle.id].append(visit) 
                            tabu_list = copy_arr_iter(plan.tabu_list)
                            tabu_list.append(visit.station.id)
                            if depth == 0:
                                new_plan = Plan(new_plan_dict, tabu_list, weight_set, branch_number)
                            else:
                                new_plan = Plan(new_plan_dict, tabu_list, plan.weight_set, plan.branch_number)

                            if next_vehicle.id == vehicle.id:
                                plans[depth+1].append(new_plan)
                            else:
                                plans[depth].append(new_plan)
            
            # Greedy construction for the rest of the route
            for plan in plans[max_depth+1]:
                dep_time = plan.next_visit.get_departure_time()
                temp_plan = Plan(plan.copy_plan(), copy_arr_iter(plan.tabu_list), plan.weight_set, plan.branch_number)
                while dep_time < end_time:
                    new_visit = self.greedy_next_visit(temp_plan, simul, 1, weight_set)
                    if new_visit != None:
                        new_visit = new_visit[0]
                        temp_plan.tabu_list.append(new_visit.station.id)
                    else:
                        break

                    temp_plan.plan[temp_plan.next_visit.vehicle.id].append(new_visit)
                    dep_time = new_visit.get_departure_time()
                    temp_plan.find_next_visit()
                completed_plans.append(temp_plan)


        plan_scores = dict() #{plan_object: list of scenario_scores}

        scenarios = self.generate_scenarioes(simul, self.number_of_scenarios)

        for plan in completed_plans:
            plan_scores[plan] = []
            for scenario_dict in scenarios:
                score = 0
                for v in plan.plan:
                    score += self.evaluate_route(plan.plan[v], scenario_dict, end_time, simul, self.evaluation_weights)    
                plan_scores[plan].append(score)
        
        return self.return_best_move(vehicle, simul, plan_scores)

    def greedy_next_visit(self, plan, simul, number_of_successors, weight_set):   #TODO: include multi-vehicle
        visits = []
        tabu_list = plan.tabu_list
        vehicle = plan.next_visit.vehicle
        
        initial_num_bikes = len(vehicle.get_bike_inventory())
        num_bikes_now = initial_num_bikes
        for visit in plan.plan[vehicle.id]:
            num_bikes_now += visit.loading_quantity
            num_bikes_now -= visit.unloading_quantity 

        potential_stations = find_potential_stations(simul, 0.15, 0.15, vehicle, num_bikes_now, tabu_list)
        if potential_stations == []: #no potential stations
            print("Lunsjpause på gutta")
            return None
        number_of_successors = min(number_of_successors, len(potential_stations))
        stations_sorted = calculate_criticality(weight_set, simul, potential_stations, plan.plan[vehicle.id]) #sorted dict {station_object: criticality_score}
        stations_sorted_list = list(stations_sorted.keys())
        next_stations = [stations_sorted_list[i] for i in range(number_of_successors)]

        for next_station in next_stations:
            arrival_time = plan.plan[vehicle.id][-1].get_departure_time() + simul.state.traveltime_vehicle_matrix[plan.plan[vehicle.id][-1].station.id][next_station.id] + settings.MINUTES_CONSTANT_PER_ACTION
            number_of_bikes_to_pick_up, number_of_bikes_to_deliver = self.calculate_loading_quantities_pilot(vehicle, num_bikes_now, simul, next_station, arrival_time)
            new_visit = Visit(next_station, number_of_bikes_to_pick_up, number_of_bikes_to_deliver, arrival_time, vehicle)
            visits.append(new_visit)
        return visits


    def evaluate_route(self, route, scenario_dict, end_time, simul, weights): #Begins with current station and loading quantities
        discounting_factors = generate_discounting_factors(len(route), 0.8) #end_factor = 1 if no discounting 
        avoided_disutility = 0
        current_time=simul.time #returns current time from the simulator in minutes, starting time for the route 
        counter=0
        for visit in route:
            avoided_violations = 0
            neighbor_roamings = 0
            improved_deviation = 0 
            
            station = visit.station
            loading_quantity = visit.loading_quantity
            unloading_quantity = visit.unloading_quantity
            neighbors = station.neighboring_stations #list of station objects

            time = visit.arrival_time
            if time > end_time:
                time = end_time
            
            initial_inventory = station.number_of_bikes()
            station_capacity = station.capacity
            net_demand = scenario_dict[station.id]   #returns net demand for the next 60 minutes from simul.time
            target_state = station.get_target_state(simul.day(), simul.hour())

            #avoided violations: 
            if net_demand>0:
                time_first_violation_no_visit = current_time+((station_capacity - initial_inventory)/net_demand)*60
            elif net_demand<0:
                time_first_violation_no_visit = current_time+(initial_inventory/(-net_demand))*60
            else:
                time_first_violation_no_visit = end_time
           
            if end_time > time_first_violation_no_visit:
                violations_no_visit = ((end_time - time_first_violation_no_visit)/60)*net_demand #negative if starvations, positive if congestions 
            else:
                violations_no_visit = 0
           
            if time > time_first_violation_no_visit:
                unavoidable_violations = ((time-time_first_violation_no_visit)/60)*net_demand
            else:
                unavoidable_violations = 0
            
            inventory_after_loading = initial_inventory + ((time-current_time)/60)*net_demand - unavoidable_violations - loading_quantity + unloading_quantity

            if net_demand>0:
                time_first_violation_after_loading = time+((station_capacity - inventory_after_loading)/net_demand)*60
            elif net_demand<0:
                time_first_violation_after_loading = time+(inventory_after_loading/(-net_demand))*60
            else:
                time_first_violation_after_loading = end_time

            
            if time_first_violation_after_loading < end_time: 
                violations_after_visit = ((end_time - time_first_violation_after_loading)/60)*net_demand
            else:
                violations_after_visit = 0 

            avoided_violations = abs(violations_no_visit) - abs(unavoidable_violations) - abs(violations_after_visit)  
            
            #improved deviation: 
            if net_demand > 0:
                ending_inventory = min(station_capacity, inventory_after_loading + ((end_time-time)/60)*net_demand)
            elif net_demand <= 0:
                ending_inventory = max(0, inventory_after_loading + ((end_time-time)/60)*net_demand)
        
            deviation_visit = abs(ending_inventory-target_state)

            if net_demand > 0:
                ending_inventory_no_visit = min(station_capacity,initial_inventory+((end_time-current_time)/60)*net_demand)
            elif net_demand <= 0:
                ending_inventory_no_visit = max(0,initial_inventory+((end_time-current_time)/60)*net_demand)
        
            deviation_no_visit = abs(ending_inventory_no_visit-target_state)

            improved_deviation = deviation_no_visit - deviation_visit

            #neighbor roamings: 
            excess_bikes = ending_inventory
            excess_locks = station_capacity-ending_inventory
            if net_demand > 0:
                excess_bikes_no_visit = min(station_capacity,initial_inventory+((end_time-current_time)/60)*net_demand)
                excess_locks_no_visit = max(0,station_capacity-(initial_inventory+((end_time-current_time)/60)*net_demand))
            elif net_demand <= 0:
                excess_bikes_no_visit = max(0,initial_inventory+((end_time-current_time)/60)*net_demand)
                excess_locks_no_visit = min(station_capacity,station_capacity-(initial_inventory+((end_time-current_time)/60)*net_demand))
            
                
            station_type, exp_num_bikes = calculate_station_type(station,net_demand,target_state)
            
            for neighbor in neighbors:
                roamings= 0
                roamings_no_visit = 0
                net_demand_neighbor = scenario_dict[neighbor.id]
                neighbor_type, exp_num_bikes_neighbor = calculate_station_type(neighbor,net_demand_neighbor,neighbor.get_target_state(simul.day(), simul.hour()))    
                if neighbor_type == station_type:
                    if net_demand_neighbor>0:
                        time_first_violation = current_time+((neighbor.capacity - neighbor.number_of_bikes())/net_demand_neighbor)*60
                    elif net_demand_neighbor<0:
                        time_first_violation = current_time+(neighbor.number_of_bikes()/(-net_demand_neighbor))*60
                    else:
                        time_first_violation = end_time
                    
                    if time_first_violation < end_time:
                        convertable_violations = (min(end_time-time_first_violation, end_time-time)/60)*net_demand_neighbor
                        if neighbor_type == 'p':
                            if abs(convertable_violations) <= excess_locks:
                                roamings+=abs(convertable_violations)
                                excess_locks-= abs(convertable_violations)
                            else:
                                roamings+=excess_locks
                                excess_locks-=excess_locks
                            
                            if abs(convertable_violations) <= excess_locks_no_visit:
                                roamings_no_visit+=abs(convertable_violations) 
                                excess_locks_no_visit-= abs(convertable_violations)
                            else:
                                roamings_no_visit+=excess_locks_no_visit
                                excess_locks_no_visit-=excess_locks_no_visit
                        
                                
                        if neighbor_type == 'd':
                            if abs(convertable_violations) <= excess_bikes:
                                roamings+=abs(convertable_violations)
                                excess_bikes-= abs(convertable_violations)
                            else:
                                roamings+=excess_bikes
                                excess_bikes-=excess_bikes

                            if abs(convertable_violations) <= excess_bikes_no_visit:
                                roamings_no_visit+=abs(convertable_violations)
                                excess_bikes_no_visit-= abs(convertable_violations)
                            else:
                                roamings_no_visit+=excess_bikes_no_visit
                                excess_bikes_no_visit-=excess_bikes_no_visit
                
                distance_scaling = ((simul.state.get_vehicle_travel_time(station.id, neighbor.id)/60)*settings.VEHICLE_SPEED)/settings.MAX_ROAMING_DISTANCE_SOLUTIONS

                neighbor_roamings += (1-distance_scaling)*(roamings-roamings_no_visit)

            # if avoided_violations < 0: #after bug fix, this still happends, but now it is because of logic in evaluate_route, unavoidable violations > violations_no_visit
            #     print("x")            

            avoided_disutility += discounting_factors[counter]*(weights[0]*avoided_violations + weights[1]*neighbor_roamings + weights[2]*improved_deviation)
        
            counter+=1
        
        return avoided_disutility 
    
    def generate_scenarioes(self, simul, number_of_scenarios):
        rng = np.random.default_rng(simul.state.seed)
        scenarios = []
        stations_dict = simul.state.stations 
        if number_of_scenarios < 2: #0 or 1, return expected net_demand values
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
                    expected_arrive_intensity = stations_dict[station_id].get_arrive_intensity(simul.day(), simul.hour())
                    arrive_intensity_stdev = stations_dict[station_id].get_arrive_intensity_stdev(simul.day(), simul.hour())
                    expected_leave_intensity = stations_dict[station_id].get_leave_intensity(simul.day(), simul.hour())
                    leave_intensity_stdev = stations_dict[station_id].get_leave_intensity_stdev(simul.day(), simul.hour())
                    
                    expected_arrive_intensity_next = stations_dict[station_id].get_arrive_intensity(simul.day(), simul.hour()+1)
                    arrive_intensity_stdev_next = stations_dict[station_id].get_arrive_intensity_stdev(simul.day(), simul.hour()+1)
                    expected_leave_intensity_next = stations_dict[station_id].get_leave_intensity(simul.day(), simul.hour()+1)
                    leave_intensity_stdev_next = stations_dict[station_id].get_leave_intensity_stdev(simul.day(), simul.hour()+1)

                    net_demand_current = rng.normal(expected_arrive_intensity, arrive_intensity_stdev) - rng.normal(expected_leave_intensity, leave_intensity_stdev)
                    net_demand_next = rng.normal(expected_arrive_intensity_next, arrive_intensity_stdev_next) - rng.normal(expected_leave_intensity_next, leave_intensity_stdev_next)

                    net_demand = (minutes_current_hour*net_demand_current + minutes_next_hour*net_demand_next)/planning_horizon
                    
                    scenario_dict[station_id] = net_demand
                scenarios.append(scenario_dict)
        return scenarios

    def return_best_move(self, vehicle, simul, plan_scores): #returns station_id 
        score_board = dict() #station id : number of times this first move returns the best solution 
        score_board_branch = dict()
        # score_board_weights = dict()  #TODO
        for scenario_id in range(self.number_of_scenarios):
            best_score = -1000
            best_plan = None
            for plan in plan_scores:
                if plan_scores[plan][scenario_id] > best_score:
                    best_plan = plan
                    best_score = plan_scores[plan][scenario_id]
            
            if best_plan == None:
                tabu_list = [vehicle2.location.id for vehicle2 in simul.state.vehicles]
                potential_stations2 = [station for station in simul.state.locations if station.id not in tabu_list]    
                rng_balanced = np.random.default_rng(None)
                print("lunsj!")
                return rng_balanced.choice(potential_stations2).id 

            best_first_move = best_plan.plan[vehicle.id][1].station.id
            if best_first_move in score_board:
                score_board[best_first_move] += 1
            else:
                score_board[best_first_move] = 1 

            if best_plan.branch_number in score_board_branch:
                score_board_branch[best_plan.branch_number] += 1
            else:
                score_board_branch[best_plan.branch_number] = 1

            # if best_plan.weight_set in score_board_weights:
            #     score_board_weights[best_plan.weight_set] += 1
            # else:
            #     score_board_weights[best_plan.weight_set] = 1
           
        score_board_sorted = dict(sorted(score_board.items(), key=lambda item: item[1], reverse=True))
        score_board_branch_sorted = dict(sorted(score_board_branch.items(), key=lambda item: item[1], reverse=True))
        # score_board_weights_sorted = dict(sorted(score_board_weights.items(), key=lambda item: item[1], reverse=True))
        print("Best branch:", list(score_board_branch_sorted.keys())[0])
        # print("Best weight:set:", list(score_board_weights_sorted.keys())[0])
        return list(score_board_sorted.keys())[0]
    

    def calculate_loading_quantities_pilot(self, vehicle, vehicle_inventory, simul, station, current_time):
        number_of_bikes_to_pick_up = 0
        number_of_bikes_to_deliver = 0 
        target_state = round(station.get_target_state(simul.day(), simul.hour()))
        net_demand = calculate_net_demand(station, simul.time, simul.day(), simul.hour(), 60)
        num_bikes_station = station.number_of_bikes() + ((current_time-simul.time)/60)*net_demand
        
        starved_neighbors=0
        congested_neighbors=0
        for neighbor in station.neighboring_stations:
            net_demand_neighbor =  calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(), 60)
            num_bikes_neighbor = neighbor.number_of_bikes() + ((current_time-simul.time)/60)*net_demand_neighbor
            if num_bikes_neighbor < 0.1*neighbor.capacity:
                starved_neighbors += 1
            elif num_bikes_neighbor > 0.9*neighbor.capacity:
                congested_neighbors += 1
        
        if num_bikes_station < target_state: #deliver bikes
            #deliver bikes, max to the target state
            number_of_bikes_to_deliver = min(vehicle_inventory, target_state - num_bikes_station + starved_neighbors)
            
        elif num_bikes_station > target_state: #pick-up bikes
            remaining_vehicle_capacity = vehicle.bike_inventory_capacity - vehicle_inventory
            number_of_bikes_to_pick_up = min(num_bikes_station - target_state + congested_neighbors, remaining_vehicle_capacity)

        return number_of_bikes_to_pick_up, number_of_bikes_to_deliver

class Visit():
    def __init__(self, station, loading_quantity, unloading_quantity, arrival_time, vehicle):
        self.station = station
        self.loading_quantity = loading_quantity #loading from station to vehicle
        self.unloading_quantity = unloading_quantity #unloading from vehicle unto station
        self.arrival_time = arrival_time #in min
        self.vehicle = vehicle 

    def get_departure_time(self):
        return self.arrival_time + (self.loading_quantity + self.unloading_quantity)*settings.MINUTES_PER_ACTION

class Plan():
    def __init__(self, copied_plan, tabu_list, weight_set=None, branch_number=None):
        self.plan = copied_plan #key: vehicle ID, value: list of vehicle route 
        self.next_visit = None     #initialize with starting station
        self.find_next_visit() 
        self.tabu_list = tabu_list
        self.weight_set = weight_set
        self.branch_number = branch_number          

    def find_next_visit(self):
        arrival_time = 1000000 
        for vehicle_id in self.plan:
            if self.plan[vehicle_id][-1].arrival_time < arrival_time:
                arrival_time = self.plan[vehicle_id][-1].arrival_time
                self.next_visit = self.plan[vehicle_id][-1]

    def copy_plan(self):
        plan_copy = dict()
        for vehicle in self.plan:
            route_copy = copy_arr_iter(self.plan[vehicle])
            plan_copy[vehicle] = route_copy
        return plan_copy


def copy_arr_iter(arr):
    root = []
    stack = [(arr, root)]
    while stack:
        (o, d), *stack = stack
        assert isinstance(o, list)
        for i in o:
            if isinstance(i, list):
                p = (i, [])
                d.append(p[1])
                stack.append(p)
            else:
                d.append(i)
    return root

    

def generate_discounting_factors(nVisits, end_factor=0.9): #number of visits, end_factor is discounting factor in final visit in route
        discounting_factors=[]
        len = nVisits
        rate = (1/end_factor)**(1/len)-1
        for visit in range(0,len):
                discount_factor = 1/((1+rate)**visit) #1 at first visit 
                discounting_factors.append(discount_factor)
        return discounting_factors