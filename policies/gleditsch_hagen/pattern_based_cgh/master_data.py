class MasterData:
    
    def __init__(self,generated_routes, all_stations,simul,omega_v,omega_d,
                 net_demand,pickup_station,deviation_not_visited, base_violations,target_state,
                 planning_horizon):
        
        
        self.generated_routes = generated_routes
        
        #calculate VIOLATIONAS, DEVIATIONS
        
        self.planning_horizon = planning_horizon
        
  
        #SETS
        
        self.S_STATIONS = []
        for station in all_stations:
            self.S_STATIONS.append(station.id)
            
        self.V_VEHICLES = set()
        for route in generated_routes:
            self.V_VEHICLES.add(route.vehicle_id)
        self.V_VEHICLES = list(self.V_VEHICLES)
        
        route_id = 0
        self.route_id_to_route = {}
        self.R_ROUTES = {v:[] for v in self.V_VEHICLES}
        for route in generated_routes:
            route.route_id = route_id
            self.route_id_to_route[route_id] = route
            self.R_ROUTES[route.vehicle_id].append(route.route_id)
            route_id += 1
        
        
        
        #PARAMETERS
        
        self.W_OMEGA_V = omega_v #TO DO
        self.W_OMEGA_D = omega_d #TO DO
        
        self.A_MATRIX = {(i,v,r):0 for i in self.S_STATIONS for v in self.V_VEHICLES for r in self.R_ROUTES[v] }
        for route in generated_routes:
            r = route.route_id
            v = route.vehicle_id
            for i in route.stations:
                self.A_MATRIX[(i,v,r)]=1    
                
        self.V_BASE = sum(base_violations[station.id] for station in all_stations)
        self.D_BASE = sum(deviation_not_visited[station.id] for station in all_stations)
                        
        self.V_PREVENTED = {(v,r):0 for v in self.V_VEHICLES for r in self.R_ROUTES[v]}
        self.D_PREVENTED = {(v,r):0 for v in self.V_VEHICLES for r in self.R_ROUTES[v]}
        #self.prevented_violations = {}
        for route in generated_routes:
            violations_prevented = 0 
            deviations_prevented = 0 
            for i in range(len(route.stations)):
                station_id = route.stations[i]                    
                # could do an if base_violation > 0:
                arrival_time = route.arrival_times[i]
                if arrival_time < self.planning_horizon: #if arrive after planning horizon, then we do not count it...    
                    num_bikes_at_visit_no_cap = (route.bikes_at_start[i] + 
                                                 net_demand[station_id]*arrival_time)
                    violation_pre = 0
                    violation_post = 0
                    
                    num_bikes_at_horizon_cap = None
                    
                    if pickup_station[station_id] == 1:
                        violation_pre = max(0,num_bikes_at_visit_no_cap-route.capacity[i])
                        bikes_after_visit = min(num_bikes_at_visit_no_cap,route.capacity[i]) - route.loading[i]
                        num_bikes_at_horizon_no_cap = bikes_after_visit + net_demand[station_id]*(
                            self.planning_horizon-arrival_time) 
                        violation_post = max(0,num_bikes_at_horizon_no_cap-route.capacity[i])
                        num_bikes_at_horizon_cap = min(route.capacity[i],num_bikes_at_horizon_no_cap)                    
                    elif pickup_station[station_id] == 0: #delivery station
                        violation_pre = max(0,- num_bikes_at_visit_no_cap)
                        bikes_after_visit = max(num_bikes_at_visit_no_cap,0) + route.unloading[i]
                        num_bikes_at_horizon_no_cap = bikes_after_visit + net_demand[station_id]*(
                            self.planning_horizon-arrival_time) 
                        violation_post = max(0,-num_bikes_at_horizon_no_cap)
                        num_bikes_at_horizon_cap = max(0,num_bikes_at_horizon_no_cap)
                    else:
                        print('PU_station: ', pickup_station[station_id])
                        raise Exception("Wrong value for pickup station")
                    violations_prevented += base_violations[station_id] - violation_pre - violation_post
                    deviation_at_horizon = abs(target_state[station_id]-num_bikes_at_horizon_cap)
                    deviations_prevented += deviation_not_visited[station_id] - deviation_at_horizon
            v = route.vehicle_id
            r = route.route_id
            self.V_PREVENTED[(v,r)] = violations_prevented
            self.D_PREVENTED[(v,r)] = deviations_prevented