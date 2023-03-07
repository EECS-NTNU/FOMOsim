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
    def __init__(self, max_depth=3, number_of_successors=3, time_horizon=60, criticality_weights=[0.2, 0.2, 0.2, 0.2, 0.2], evaluation_weights=[0.33, 0.33, 0.33]):
        self.max_depth = max_depth
        self.number_of_successors = number_of_successors
        self.time_horizon = time_horizon
        self.criticality_weights = criticality_weights
        self.evaluation_weights = evaluation_weights
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
        
        route = []
        route.append(Visit(vehicle.location, number_of_bikes_to_pick_up, number_of_bikes_to_deliver, simul.time, vehicle))

        ##########################################
        #               WHERE TO GO              #
        ##########################################

        next_station = self.PILOT_function(simul, vehicle, route, self.max_depth, self.number_of_successors, end_time)
        
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


    def greedy_next_visit(self, route, vehicle, simul, number_of_successors):   #TODO: include multi-vehicle
        visits = []
        tabu_list = [vehicle2.location.id for vehicle2 in simul.state.vehicles] #do not go where other vehicles are (going)
        stations_in_route = [visit.station.id for visit in route] #only visit a station maximum once during a route horizon 
        tabu_list.extend(stations_in_route)
        
        num_bikes_vehicle = len(vehicle.get_bike_inventory())
        potential_stations = find_potential_stations(simul, 0.15, 0.15, vehicle, num_bikes_vehicle, tabu_list)
        if potential_stations == []: #no potential stations
            print("Lunsjpause på gutta")
            return None
        number_of_successors = min(number_of_successors, len(potential_stations))
        stations_sorted = calculate_criticality(self.criticality_weights, simul, potential_stations, route) #sorted dict {station_object: criticality_score}
        stations_sorted_list = list(stations_sorted.keys())
        next_stations = [stations_sorted_list[i] for i in range(number_of_successors)]

        
        for visit in route:
            num_bikes_vehicle = num_bikes_vehicle + visit.loading_quantity - visit.unloading_quantity

        for next_station in next_stations:
            arrival_time = route[-1].get_departure_time() + simul.state.traveltime_vehicle_matrix[route[-1].station.id][next_station.id] + settings.MINUTES_CONSTANT_PER_ACTION
            number_of_bikes_to_pick_up, number_of_bikes_to_deliver = self.calculate_loading_quantities_pilot(vehicle, num_bikes_vehicle, simul, next_station, arrival_time)
            visits.append(Visit(next_station, number_of_bikes_to_pick_up, number_of_bikes_to_deliver, arrival_time, vehicle))
        return visits


    def evaluate_route(self, route, demand_scenario, end_time, simul, weights): #Begins with current station and loading quantities
        discounting_factors = generate_discounting_factors(len(route), 0.8) #end_factor = 1 if no discounting 
        avoided_disutility = 0
        current_time=simul.time #returns current time from the simulator in minutes, starting time for the route 
        time = current_time 
        previous_station=None
        counter=0
        for visit in route:
            avoided_violations = 0
            neighbor_roamings = 0
            improved_deviation = 0
            
            station = visit.station
            loading_quantity = visit.loading_quantity
            unloading_quantity = visit.unloading_quantity
            neighbors = station.neighboring_stations #list of station objects

            if previous_station != None:
                time = visit.arrival_time
            else: #we are on the first station 
                time = time
            
            initial_inventory = station.number_of_bikes()
            station_capacity = station.capacity
            net_demand = calculate_net_demand(station,current_time,simul.day(),simul.hour(), 60) #calculates the net demand for the next 60 minutes from the current simulation time. Can be adjusted to return a value based on scenario  
            target_state = station.get_target_state(simul.day(), simul.hour())

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
                net_demand_neighbor = calculate_net_demand(neighbor,current_time,simul.day(),simul.hour(), 60)
                neighbor_type, exp_num_bikes_neighbor = calculate_station_type(neighbor,net_demand_neighbor,neighbor.get_target_state(simul.day(), simul.hour()))    
                if neighbor_type == station_type:
                    if net_demand_neighbor>0:
                        time_first_violation = current_time+((neighbor.capacity - neighbor.number_of_bikes())/net_demand_neighbor)*60
                    elif net_demand_neighbor<0:
                        time_first_violation = current_time+(neighbor.number_of_bikes()/(-net_demand_neighbor))*60
                    else:
                        time_first_violation = end_time
                    
                    if time_first_violation < end_time:
                        violations = (min(end_time-time_first_violation, end_time-time)/60)*net_demand_neighbor
                        if neighbor_type == 'p':
                            if abs(violations) <= excess_locks:
                                roamings+=abs(violations)
                                excess_locks-= abs(violations)
                            else:
                                roamings+=excess_locks
                                excess_locks-=excess_locks
                            
                            if abs(violations) <= excess_locks_no_visit:
                                roamings_no_visit+=abs(violations) 
                                excess_locks_no_visit-= abs(violations)
                            else:
                                roamings_no_visit+=excess_locks_no_visit
                                excess_locks_no_visit-=excess_locks_no_visit
                        
                                
                        if neighbor_type == 'd':
                            if abs(violations) <= excess_bikes:
                                roamings+=abs(violations)
                                excess_bikes-= abs(violations)
                            else:
                                roamings+=excess_bikes
                                excess_bikes-=excess_bikes

                            if abs(violations) <= excess_bikes_no_visit:
                                roamings_no_visit+=abs(violations)
                                excess_bikes_no_visit-= abs(violations)
                            else:
                                roamings_no_visit+=excess_bikes_no_visit
                                excess_bikes_no_visit-=excess_bikes_no_visit
                
                distance_scaling = ((simul.state.get_vehicle_travel_time(station.id, neighbor.id)/60)*settings.VEHICLE_SPEED)/settings.MAX_ROAMING_DISTANCE_SOLUTIONS

                neighbor_roamings += (1-distance_scaling)*(roamings-roamings_no_visit)
           

            avoided_disutility += discounting_factors[counter]*(weights[0]*avoided_violations + weights[1]*neighbor_roamings + weights[2]*improved_deviation)
            
            #for next iteration:
            time = visit.get_departure_time() #the time after the loading operations are done 
            previous_station = station
            counter+=1
        
        return avoided_disutility 
    
    
    def calculate_loading_quantities_pilot(self, vehicle, vehicle_inventory, simul, station, current_time):
        
        number_of_bikes_to_pick_up = 0
        number_of_bikes_to_deliver = 0
        target_state = round(station.get_target_state(simul.day(), int((current_time//60)%24)))  #assume same day
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