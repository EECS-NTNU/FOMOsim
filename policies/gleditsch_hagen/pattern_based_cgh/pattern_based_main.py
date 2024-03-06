import numpy as np
import time
import copy

import settings

from ..utils import calculate_net_demand, calculate_time_to_violation, extract_N_best_elements
from ...criticality_scores import calculate_criticality_normalized
from .Routes import Route
from .master_data import MasterData
from .master_model import run_master_model

class PatternBasedCGH:
    def __init__(self, state,parameters,vehicle,other_vehicle_at_same_location,
                 num_scoring_iterations=0   #NOT YET IMPLEMENTED, so put to zero. Also, not much effect.
                 ):
        self.branching_constant = parameters["branching_constant"]
        [self.omega1,self.omega2,self.omega3,self.omega4] = parameters["crit_weights"]
        [self.omega_v,self.omega_d]= parameters["vd_weights"]
        self.threshold_service_vehicle = parameters["threshold_service_vehicle"]
        self.num_scoring_iterations = num_scoring_iterations
        
        self.state = state
        self.vehicle = vehicle
        self.planning_horizon = parameters["planning_horizon"]
        self.other_vehicle_at_same_location = other_vehicle_at_same_location
        
        self.data = None
        self.master_solution = None
        
        #first do some data preprocessing. These calculations are necessary for several stages in the cgh

        self.all_stations = state.locations 
        self.pickup_stations = []
        self.delivery_stations = []
        
        self.initial_routes = []
           
        self.net_demand = {station.id:0 for station in self.all_stations}
        self.pickup_station = {station.id:0 for station in self.all_stations}
        self.deviation_now = {station.id:0 for station in self.all_stations}
        self.deviation_not_visited = {station.id:0 for station in self.all_stations}
        self.time_to_violation = {station.id:0 for station in self.all_stations}
        self.base_violations = {station.id:0 for station in self.all_stations}
        self.target_state = {station.id:0 for station in self.all_stations}
        
        for station in self.all_stations:
            self.net_demand[station.id] = round(calculate_net_demand(station,self.state.time,self.state.day(),
                                    self.state.hour(),self.planning_horizon),4)
            num_bikes_not_visited_no_cap = station.number_of_bikes() + self.net_demand[station.id]*self.planning_horizon
    
            if self.net_demand[station.id] >= 0:
                self.pickup_station[station.id] = 1
                self.pickup_stations.append(station)
                self.base_violations[station.id] = max(0,num_bikes_not_visited_no_cap-station.capacity)
                num_bikes_not_visited_cap = min(num_bikes_not_visited_no_cap,station.capacity)
            else:
                self.pickup_station[station.id] = 0
                self.delivery_stations.append(station)
                self.base_violations[station.id] = max(0,- num_bikes_not_visited_no_cap)
                num_bikes_not_visited_cap = max(num_bikes_not_visited_no_cap,0) 
    
            self.time_to_violation[station.id] = calculate_time_to_violation(self.net_demand[station.id], station)
            
            self.target_state[station.id] = station.get_target_state() # TO DO, there is a mismatch between the target state at the end of planning horizon, and target state at end of hour!!
            self.deviation_not_visited[station.id] = round(abs(num_bikes_not_visited_cap-self.target_state[station.id]),4)
            self.deviation_now[station.id] = round(abs(station.number_of_bikes()-self.target_state[station.id]),4)

        self.main()

    def main(self):
        #Initialization
        start= time.time()  
        self.initial_routes = self.initialization()    #GENERATE SET OF ROUTES
        self.duration_route_initialization = time.time() - start

#testing
# for i in range(len(self.initial_routes)):
#     print('Route: ', i)
#     print('Vehicle: ', self.initial_routes[i].vehicle.id)
#     print('Stations: ', [station.id for station in self.initial_routes[i].stations])
#     print('Arrival times: ', self.initial_routes[i].arrival_times)
#     print('loading: ', self.initial_routes[i].loading)
#     print('unloading: ', self.initial_routes[i].unloading)
#     print('vehicle_level: ', self.initial_routes[i].vehicle_level)
#     print('-------------------------------------------')        
        #Master
        start= time.time() 
        self.master_problem(self.initial_routes)
        self.duration_master_problem_first = time.time() - start
        
        #Scoring/pricing   WE DROP THIS FOR NOW!!!! num_scoring_iterations = 0
        for i in range(self.num_scoring_iterations):
            self.scoring_problem()
            self.master_problem()
           
    
    def initialization(self):
        
        initial_routes = [] 
        
        #Then, perform the algorithm
        if self.other_vehicle_at_same_location: #only do planning for a single vehicle
            initial_routes = self.route_extension_algo(self.vehicle)
        else:
            for vehicle in self.state.vehicles:  #NOTE, WE GET MANY SIMILAR ROUTES!!! DRIVING TIME/DISTANCE IS NEGLIGABLE..??  Not true...
                generated_routes = self.route_extension_algo(vehicle)
                initial_routes += generated_routes
        
        return initial_routes          
            

    def route_extension_algo(self,vehicle):  #do the extension for all vehicles, not only the trigger
        
        trigger=False
        if vehicle.id==self.vehicle.id:
            trigger=True
        #initiate the route
        
        finished_routes = list()
        start_of_route = Route(vehicle.id,vehicle.eta, vehicle.location.id, vehicle.location.number_of_bikes(),
                               vehicle.location.capacity, vehicle.get_bike_inventory(),
                               vehicle.bike_inventory_capacity,vehicle.parking_time,vehicle.handling_time,
                               trigger,self.state.time,self.state.day(),self.state.hour(),self.planning_horizon,
                               self.pickup_station[vehicle.location.id],self.net_demand[vehicle.location.id])
        partial_routes = [start_of_route]
        while len(partial_routes) > 0 :
            pr = partial_routes.pop(0)
            pr.loading_algorithm(self.state.stations) #both loads and arrival times
            if len(pr.stations)==2:
                finished_routes.append(pr)   #FOR FEASIBILITY PURPOSES,
            if pr.duration_route < self.planning_horizon:
                cs = pr.stations[pr.num_visits-1] #current station
                potential_stations = self.filtering(pr)
                driving_times = [self.state.get_vehicle_travel_time(cs,ps2.id) for ps2 in potential_stations]
                criticalities = []
                for i in range(len(potential_stations)):
                    ps = potential_stations[i].id #potential station
                    #criticalities.append(self.calculate_criticality(cs,ps.id))    #
                    criticalities.append(calculate_criticality_normalized(
                        weights=[self.omega1,self.omega2,self.omega3,self.omega4],
                        state=self.state,
                        start_station_id = cs, 
                        potential_station_id = ps,
                        net_demand_ps = self.net_demand[ps],
                        max_net_demand = max([abs(x) for x in list(self.net_demand.values())]),
                        max_driving_time = max(driving_times) ,
                        max_time_to_violation = max(list(self.time_to_violation.values())),
                        max_deviation = max(list(self.deviation_now.values())),   # OR USE DEVIATION NOT VISITED
                        #planning_horizon=60
                    ))
                indices_best_extensions = extract_N_best_elements(criticalities,self.branching_constant)
                for i in indices_best_extensions:
                    pr2 = copy.deepcopy(pr) # TO DO: DEN HER ER SUPER TREIG. jeg har fjernet simul fra Route class, men burde kanskje fjerne mere.
                    driving_time_to_new_station = self.state.get_vehicle_travel_time(
                        pr.stations[pr.num_visits-1],potential_stations[i].id)
                    pr2.add_station(potential_stations[i].id,driving_time_to_new_station,
                                    self.pickup_station[potential_stations[i].id],
                                    self.net_demand[potential_stations[i].id],
                                    potential_stations[i].number_of_bikes(),
                                    potential_stations[i].capacity
                                    )   
                    partial_routes.append(pr2)
            else:
                finished_routes.append(pr)
        return finished_routes
            
                
    def filtering(self,route):
        next_stations = self.all_stations
        if route.num_visits == 1:
            if route.pickup_station[0] == 1:
                if len(route.vehicle_initial_inventory) >= (1-self.threshold_service_vehicle) * route.vehicle_cap:
                    next_stations=self.delivery_stations
            else:  #so, DELIVERY_STATION
                if len(route.vehicle_initial_inventory) <= self.threshold_service_vehicle * route.vehicle_cap:
                    next_stations = self.pickup_stations
        else: #so at least two visits
            if (route.pickup_station[route.num_visits-1]==1) and (route.pickup_station[route.num_visits-2]==1):
                next_stations=self.delivery_stations
            elif (route.pickup_station[route.num_visits-1]==0) and (route.pickup_station[route.num_visits-2]==0):
                next_stations=self.pickup_stations
        #remove existing stations. IF NOT DOING THIS, THEN WE GET INFEASIBILITIES!!! TO DO: check
        # I believe that it might happen that a station gets assigned to a route, whereas it would have been better in another route.
        filtered_stations = list(set(next_stations)-set(route.stations))  
        return filtered_stations    
            
    def master_problem(self,input_routes):
        
        self.data = MasterData(input_routes,self.all_stations,self.omega_v,self.omega_d,
                               self.net_demand,self.pickup_station,self.deviation_not_visited,
                               self.base_violations,self.target_state,self.planning_horizon)
        self.master_solution = run_master_model(self.data)
        
        
    def return_solution(self, vehicle_index_input):
        loading = 0
        unloading = 0
        for var in self.master_solution.getVars():
            name = var.varName.strip("]").split("[")
            indices_vr = name[1].split(',')
            vehicle_index = indices_vr[0]
            route_index = indices_vr[1]
            round_val = round(var.x, 0)
            if round_val == 1:
                #print('variable equaling one: ', var)
                #print('vehicle index: ',int(vehicle_index))
                if name[0] == 'l_lambda' and int(vehicle_index) == vehicle_index_input:
                    route = self.data.route_id_to_route[int(route_index)]
                    loading = route.loading[0]
                    unloading = route.unloading[0]
                    station_id = route.stations[1]
        return station_id, loading, unloading







    # THE FOLLOWING CAN BE REMOVED:
    def calculate_criticality(self,start_station_id, potential_station_id):      #OLD!!!
        driving_time = self.state.get_vehicle_travel_time(start_station_id,potential_station_id)        
        return round((
            - self.omega1*self.time_to_violation[potential_station_id] 
            + self.omega2*abs(self.net_demand[potential_station_id])    #THIS IS NEW, NOT SURE IF IT WAS IMPLEMENTED RIGHT. the paper is also not clear
            - self.omega3*driving_time 
            + self.omega4*self.deviation_not_visited[potential_station_id]),3)





