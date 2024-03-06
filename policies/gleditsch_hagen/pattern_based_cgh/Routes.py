# -*- coding: utf-8 -*-

import time
import settings

from ..utils import calculate_net_demand

class Route:  # OR COLUMNS

    def __init__(self, vehicle_id, eta, starting_station_id,station_load_start,station_cap,vehicle_initial_inventory,vehicle_cap,
                 vehicle_parking_time, vehicle_handling_time,
                 trigger,time,day,hour,planning_horizon,pickup_station,net_demand):
        
        self.vehicle_id = vehicle_id
        self.eta = eta
        self.starting_station_id = starting_station_id
        self.vehicle_initial_inventory = vehicle_initial_inventory
        self.vehicle_cap = vehicle_cap
        self.vehicle_parking_time =vehicle_parking_time
        self.vehicle_handling_time = vehicle_handling_time
        
        self.trigger = trigger
        self.time = time
        self.day = day
        self.hour = hour
        self.planning_horizon = planning_horizon
        
        self.route_id = -1
        
        self.stations = []
        self.pickup_station = []
        self.net_demand = []
        self.loading = []  
        self.unloading = []
        self.arrival_times = []
        self.vehicle_level = []
        self.station_loads_at_visit = []
        self.regret = []
        self.travel_times = []
        self.bikes_at_start = []
        self.capacity = [station_cap] #"station cap"
        
        self.num_visits = 1
        self.first_arrival_time = 0
        if not self.trigger: # if vehicle is still underway (loading or driving)
            if self.eta > 0: 
                self.first_arrival_time = round(self.eta - self.time,3)
        self.duration_route = self.first_arrival_time #WHETHER A ROUTE IS EXTENDED DEPENDS ON THE DURATION!! 
        
        self.station_load_first_visit = station_load_start+net_demand*self.first_arrival_time
        
        self.stations.append(starting_station_id)
        self.pickup_station.append(pickup_station)
        self.bikes_at_start.append(station_load_start)
        self.net_demand.append(net_demand)
        self.arrival_times.append(round(self.first_arrival_time,3))
        
        
        

    
        
    def loading_algorithm(self,stations):  #maybe we do not need to send all simul info just yet    TO DO, check how much time this step takes
        #THE PAPER IS LOCK-CENTRIC. ONLY ONE SMALL TYPO IN PROBLEM DESCRIPTION
        #    
        #Positive net demand: PICKUP STATION
        #Negative net demand: DELIVERY STATION
        
        start_time = time.time()
        
        #SOME OF THIS CAN BE IN THE OVERARCHING CLASS
        self.num_visits = len(self.stations)
        
        #this has to be recalculated when the loading algorithm is called
        self.loading = [0 for i in range(self.num_visits)]  #perspective from VEHICLE
        self.unloading = [0 for i in range(self.num_visits)]
        self.arrival_times = [None for i in range(self.num_visits)]
        self.arrival_times[0] = self.first_arrival_time  #length/duration of the route
        self.vehicle_level = [0 for i in range(self.num_visits)]  #when arriving at the station
        
        self.vehicle_level[0] = len(self.vehicle_initial_inventory)
        self.station_loads_at_visit = [0 for i in range(self.num_visits)]
        self.station_loads_at_visit[0] = self.station_load_first_visit
        self.regret = [0 for i in range(self.num_visits)]

        
        ########    
        # MAIN #
        ########
        
        i = 0    #NOTE THAT WE MOST LIKELY CAN START TWO STEPS AWAY FROM THE 
        spare_cap=0
        while i <= self.num_visits-1:    
            
            #DETERMINE Q_HALF
            Q_half = self.vehicle_cap
            if i < (len(self.stations)-1):  #if not last station visit in route
                if self.pickup_station[i]-self.pickup_station[i+1]==0:  #same type of station
                    Q_half = 0.5*self.vehicle_cap + spare_cap*self.regret[i+1]
            
             
            #DETERMINE THE (UN)LOADING
            if self.pickup_station[i] == 1: #pickup_station / S^L
                vehicle_spare_capacity = self.vehicle_cap-self.vehicle_level[i]   
                self.loading[i] = min(vehicle_spare_capacity,self.station_loads_at_visit[i],Q_half)
                if i >= 1:
                    if self.pickup_station[i-1] == 1:
                        spare_cap = vehicle_spare_capacity - self.loading[i]
                        if spare_cap > 0:  #REMAINING CAPACITY AFTER PICK UP 
                            if  self.station_loads_at_visit[i-1] - self.loading[i-1] > 0:  #COULD HAVE PICKED UP MORE IN PREVIOUS
                                if self.regret[i] == 0: # not yet performed a regret    
                                    self.regret[i] = 1    
                                    i = i-1
                                    continue #break this iteration and go a step back
            else: #delivery_station / S^U
                self.unloading[i] = min(stations[self.stations[i]].capacity-self.station_loads_at_visit[i],self.vehicle_level[i],Q_half)
                if i >= 1:
                    if self.pickup_station[i-1] == 0:
                        spare_cap = self.vehicle_level[i] - self.unloading[i]
                        if  spare_cap > 0:  #COULD HAVE UNLOADED MORE 
                            if  stations[self.stations[i-1]].capacity-self.station_loads_at_visit[i-1] + self.unloading[i-1] > 0:  #CAN UNLOAD MORE
                                if self.regret[i] == 0: # not yet performed a regret
                                    self.regret[i] = 1
                                    i = i-1
                                    continue
            
            if i <= self.num_visits-2:   #if we are not yet at the last station in the route, then calculate the following
                self.arrival_times[i+1] = round((self.arrival_times[i]+self.vehicle_parking_time+
                                      self.vehicle_handling_time*(self.loading[i]+self.unloading[i])+
                                      self.travel_times[i]),3)                 
                self.vehicle_level[i+1] = self.vehicle_level[i] + self.loading[i] - self.unloading[i]

                load_at_visit_no_cap = stations[self.stations[i+1]].number_of_bikes()+self.net_demand[i+1]*self.arrival_times[i+1]  #TO DO: should do rounding here?
                if load_at_visit_no_cap < 0:
                    self.station_loads_at_visit[i+1] = 0
                elif load_at_visit_no_cap > stations[self.stations[i+1]].capacity:
                    self.station_loads_at_visit[i+1] = stations[self.stations[i+1]].capacity
                else:
                    self.station_loads_at_visit[i+1] = load_at_visit_no_cap
                
            i = i+1
        
        j = self.num_visits-1
        self.duration_route = self.arrival_times[j]+self.vehicle_handling_time*(self.loading[j]+self.unloading[j])
        
        self.time_used_in_algo =time.time()-start_time
        
    def add_station(self, station_id, travel_time,pickup_station,net_demand,station_load_start,capacity):
        self.stations.append(station_id)    
        self.travel_times.append(travel_time)
        self.pickup_station.append(pickup_station)
        self.net_demand.append(net_demand)
        self.bikes_at_start.append(station_load_start)
        self.capacity.append(capacity)
        
    def loading_algorithm2(self,route,planning_horizon):       #START TWO STEPS BACK; SAFE MANY CALCULATIONS TO DO
        pass
