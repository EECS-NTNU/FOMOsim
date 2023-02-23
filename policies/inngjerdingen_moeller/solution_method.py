'''rename this module and class eg., PILOT, InngjerdingenMoellerPolicy_2, heuristic or something similar '''
from policies import Policy
import sim
from criticality_score_neighbor import calculate_criticality, calculate_station_type
import settings
from policies.gleditsch_hagen.utils import calculate_net_demand
from greedy_policy_with_neighbors import calculate_loading_quantities

class SolutionMethod(Policy):
    def __init__(self):
        super().__init__()

    def get_best_action(self, simul, vehicle):
        next_station = None
        bikes_to_pickup = []
        bikes_to_deliver = []  
        
        #########################################
        #               WHAT TO DO              #
        #########################################
        num_bikes_vehicle = len(vehicle.get_bike_inventory())
        bikes_to_pickup, bikes_to_deliver = calculate_loading_quantities(vehicle, simul, vehicle.location)
        number_of_bikes_to_pick_up = len(bikes_to_pickup)
        number_of_bikes_to_deliver = len(bikes_to_deliver)
        
        bikes_at_vehicle_after_rebalancing = num_bikes_vehicle + number_of_bikes_to_pick_up - number_of_bikes_to_deliver

        ##########################################
        #               WHERE TO GO              #
        ##########################################

        next_station = self.PILOT_function()
        
        return sim.Action(
            [],               # batteries to swap
            bikes_to_pickup, #list of bike id's
            bikes_to_deliver, #list of bike id's
            next_station, #id
        )   


    def PILOT_function(self):
        #calls the greedy function recursively + smart filtering, branching and depth
        return None
    
    def greedy_with_neighbor():
        #calls calculate_criticality 
        #should calculate_criticality recieve potential stations or receive
        # all stations and make the filtering inside?
        return None

    def evaluate_route(self, route, demand_scenario, time_horizon, simul, weights): #a route can be a list with (station_object, loading_quantity)-tuples as list elements. Begins with current station and loading quantities
        avoided_disutility = 0
        current_time=simul.time #returns current time from the simulator in minutes, starting time for the route 
        end_time = current_time + time_horizon
        time = current_time 
        previous_station=None
        for action in route:
            avoided_violations = 0
            neighbor_roamings = 0
            improved_deviation = 0
            
            station = action[0]
            loading_quantity = action[1] #positive loading quantity means unloading of bicycles to the station 
            neighbors = station.neighboring_stations #list of station objects

            if previous_station != None:
                time += simul.state.get_vehicle_travel_time(previous_station.id, station.id) #arrival time at the station
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
            
            inventory_after_loading = initial_inventory + ((time-current_time)/60)*net_demand - unavoidable_violations + loading_quantity

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
                
                distance_scaling = ((simul.state.get_vehicle_travel_time(station.id, neighbor.id)/60)*settings.VEHICLE_SPEED)/settings.MAX_ROAMING_DISTANCE

                neighbor_roamings += (1-distance_scaling)*(roamings-roamings_no_visit)
           

            avoided_disutility += (weights[0]*avoided_violations + weights[1]*neighbor_roamings + weights[2]*improved_deviation)
            
            #for next iteration:
            time += abs(loading_quantity*settings.MINUTES_PER_ACTION) #the time after the loading operations are done 
            previous_station = station
        
        return avoided_disutility 