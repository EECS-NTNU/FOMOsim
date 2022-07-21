# -*- coding: utf-8 -*-
from policies.gleditsch_hagen.utils import calculate_net_demand

class Route:  # OR COLUMNS

    def __init__(self, vehicle, trigger,time,day,hour,planning_horizon):
        self.vehicle = vehicle
        self.trigger = trigger
        self.time = time
        self.day = day
        self.hour = hour
        self.planning_horizon = planning_horizon
        self.route_id = -1
        
        self.stations = []
        self.loading = []  
        self.unloading = []
        self.arrival_times = []
        self.vehicle_level = []
        self.station_loads_at_visit = []
        self.regret = []
        self.pickup_station = []
        self.duration_route = 0 #WHETHER A ROUTE IS EXTENDED DEPENDS ON THE DURATION!!    
        
        self.starting_station = vehicle.location   #Necessary?
        self.stations.append(vehicle.location)
        self.num_visits = 1
        self.first_arrival_time = 0
        if not self.trigger: # if vehicle is still underway (loading or driving)
            if self.vehicle.eta == 0: #It equals zero when the simulation starts
                self.arrival_times[0] = 0 
            elif self.vehicle.eta > 0: 
                self.first_arrival_time = self.vehicle.eta - self.time 
        net_demand = calculate_net_demand(self.stations[0],self.time,self.day,self.hour,self.planning_horizon)
        self.station_load_first_visit = self.stations[0].number_of_scooters()+net_demand*self.first_arrival_time
        

    def add_station(self, station):
        self.stations.append(station)
        
    def loading_algorithm(self):  #maybe we do not need to send all simul info just yet    TO DO, check how much time this step takes
        #THE PAPER IS LOCK-CENTRIC. ONLY ONE SMALL TYPO IN PROBLEM DESCRIPTION
        #    
        #Positive net demand: PICKUP STATION
        #Negative net demand: DELIVERY STATION
        
        #SOME OF THIS CAN BE IN THE OVERARCHING CLASS
        self.num_visits = len(self.stations)
        self.loading = [0 for i in range(self.num_visits)]  #perspective from VEHICLE
        self.unloading = [0 for i in range(self.num_visits)]
        self.arrival_times = [None for i in range(self.num_visits)]
        self.arrival_times[0] = self.first_arrival_time  #length/duration of the route
        self.vehicle_level = [0 for i in range(self.num_visits)]  #when arriving at the station
        self.vehicle_level[0] = len(self.vehicle.get_scooter_inventory())
        self.station_loads_at_visit = [0 for i in range(self.num_visits)]
        self.station_loads_at_visit[0] = self.station_load_first_visit
        self.regret = [0 for i in range(self.num_visits)]
        self.pickup_station = [0 for i in range(self.num_visits)]
        for i in range(len(self.stations)):
            net_demand = calculate_net_demand(self.stations[i],self.time,self.day,self.hour,self.planning_horizon)    
            if net_demand >= 0: #pickup_station / S^L
                 self.pickup_station[i] = 1  
        
        ########    
        # MAIN #
        ########
        
        i = 0    #NOTE THAT WE MOST LIKELY CAN START TWO STEPS AWAY FROM THE 
        spare_cap=0
        while i <= self.num_visits-1:    
            
            #DETERMINE Q_HALF
            Q_half = self.vehicle.scooter_inventory_capacity
            if i < (len(self.stations)-1):  #if not last station visit in route
                if self.pickup_station[i]-self.pickup_station[i+1]==0:  #same type of station
                    Q_half = 0.5*self.vehicle.scooter_inventory_capacity + spare_cap*self.regret[i+1]
            
             
            #DETERMINE THE (UN)LOADING
            if self.pickup_station[i] == 1: #pickup_station / S^L
                vehicle_spare_capacity = self.vehicle.scooter_inventory_capacity-self.vehicle_level[i]   
                self.loading[i] = min(vehicle_spare_capacity,self.station_loads_at_visit[i],Q_half)
                if i >= 1:
                    if self.pickup_station[i-1] == 1:
                        spare_cap = vehicle_spare_capacity - self.loading
                        if spare_cap > 0:  #REMAINING CAPACITY AFTER PICK UP 
                            if  self.station_loads_at_visit[i-1] - self.loading[i-1] > 0:  #COULD HAVE PICKED UP MORE IN PREVIOUS
                                if self.regret[i] == 0: # not yet performed a regret    
                                    self.regret[i] = 1    
                                    i = i-1
                                    continue #break this iteration and go a step back
            else: #delivery_station / S^U
                self.unloading[i] = min(self.stations[i].capacity-self.station_loads_at_visit[i],self.vehicle_level[i],Q_half)
                if i >= 1:
                    if self.pickup_station[i-1] == 0:
                        spare_cap = self.vehicle_level[i] - self.unloading
                        if  spare_cap > 0:  #COULD HAVE UNLOADED MORE 
                            if  self.stations[i-1].capacity-self.station_loads_at_visit[i-1] + self.unloading[i-1] > 0:  #CAN UNLOAD MORE
                                if self.regret[i] == 0: # not yet performed a regret
                                    self.regret[i] = 1
                                    i = i-1
                                    continue
            if i <= self.num_visits-2:  
                self.arrival_times[i+1] = (self.arrival_times[i]+self.vehicle.parking_time+
                                      self.vehicle.handling_time*(self.loading[i]+self.unloading[i])+
                                      self.stations[i].distance_to(self.stations[i+1].get_location())/self.vehicle.speed)                  
                self.vehicle_level[i+1] = self.vehicle_level[i] + self.loading[i] - self.unloading[i]
                self.station_loads_at_visit[i+1] = self.station_loads_at_visit[i] - self.loading[i] + self.unloading[i] 
            
            i = i+1
        
        j = self.num_visits-1
        self.duration_route = self.arrival_times[j]+self.vehicle.handling_time*(self.loading[j]+self.unloading[j])
        
    def loading_algorithm2(self,route,simul,planning_horizon):       #START TWO STEPS BACK; SAFE MANY CALCULATIONS TO DO
        pass
