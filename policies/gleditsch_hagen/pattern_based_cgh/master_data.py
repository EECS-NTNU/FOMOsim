class MasterData:
    
    def __init__(self,generated_routes,all_stations,simul,omega_v,omega_d,
                 net_demand,pickup_station,deviation_not_visited, base_violations,target_state,
                 planning_horizon):
        
        #calculate VIOLATIONAS, DEVIATIONS
        
        self.planning_horizon = planning_horizon
        
  
        #SETS
        
        self.S_STATIONS = []
        for station in all_stations:
            self.S_STATIONS.append(station.id)
            
        self.V_VEHICLES = []
        for vehicle in simul.state.vehicles:
            self.V_VEHICLES.append(vehicle.id)
        
        route_id = 0
        self.route_id_to_route = {}
        self.R_ROUTES = {v:[] for v in self.V_VEHICLES}
        for route in generated_routes:
            route.route_id = route_id
            self.route_id_to_route[route_id] = route
            vehicle = route.vehicle
            self.R_ROUTES[vehicle.id].append(route.route_id)
            route_id += 1
        
        #PARAMETERS
        
        self.W_OMEGA_V = omega_v #TO DO
        self.W_OMEGA_D = omega_d #TO DO
        
        self.A_MATRIX = {(i,v,r):0 for i in self.S_STATIONS for v in self.V_VEHICLES for r in self.R_ROUTES[v] }
        for route in generated_routes:
            r = route.route_id
            v = route.vehicle.id
            for station in route.stations:
                i = station.id
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
                station = route.stations[i]                    
                # could do an if base_violation > 0:
                arrival_time = route.arrival_times[i]
                if arrival_time < self.planning_horizon: #if arrive after planning horizon, then we do not count it...    
                    num_bikes_at_visit_no_cap = (station.number_of_bikes() + 
                                                 net_demand[station.id]*arrival_time)
                    violation_pre = 0
                    violation_post = 0
                    
                    num_bikes_at_horizon_cap = None
                    
                    if pickup_station[station.id] == 1:
                        violation_pre = max(0,num_bikes_at_visit_no_cap-station.capacity)
                        bikes_after_visit = min(num_bikes_at_visit_no_cap,station.capacity) - route.loading[i]
                        num_bikes_at_horizon_no_cap = bikes_after_visit + net_demand[station.id]*(
                            self.planning_horizon-arrival_time) 
                        violation_post = max(0,num_bikes_at_horizon_no_cap-station.capacity)
                        num_bikes_at_horizon_cap = min(station.capacity,num_bikes_at_horizon_no_cap)                    
                    elif pickup_station[station.id] == 0: #delivery station
                        violation_pre = max(0,- num_bikes_at_visit_no_cap)
                        bikes_after_visit = max(num_bikes_at_visit_no_cap,0) + route.unloading[i]
                        num_bikes_at_horizon_no_cap = bikes_after_visit + net_demand[station.id]*(
                            self.planning_horizon-arrival_time) 
                        violation_post = max(0,-num_bikes_at_horizon_no_cap)
                        num_bikes_at_horizon_cap = max(0,num_bikes_at_horizon_no_cap)
                    else:
                        print('PU_station: ', pickup_station[station.id])
                        raise Exception("Wrong value for pickup station")
                    violations_prevented += base_violations[station.id] - violation_pre - violation_post
                    deviation_at_horizon = abs(target_state[station.id]-num_bikes_at_horizon_cap)
                    deviations_prevented += deviation_not_visited[station.id] - deviation_at_horizon
            v = route.vehicle.id
            r = route.route_id
            self.V_PREVENTED[(v,r)] = violations_prevented
            self.D_PREVENTED[(v,r)] = deviations_prevented