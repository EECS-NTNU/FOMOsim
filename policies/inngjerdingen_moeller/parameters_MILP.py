
#from policies.gleditsch_hagen.utils import calculate_net_demand


####################################
# set the right working directory #
###################################

import os 
import sys
from pathlib import Path
import random

path = Path(__file__).parents[2]        # The path seems to be correct either way, sys.path.insert makes the difference
os.chdir(path)
# print(os. getcwd())
#path = "../../sim" #gått opp to mapper
#path = "./adasd/Asdads" #./ samme mappe
sys.path.insert(0, '') #make sure the modules are found in the new working directory

##############################################################


import sim
import policies
from settings import BIKE_SPEED, DEFAULT_DEPOT_CAPACITY, MINUTES_CONSTANT_PER_ACTION, MINUTES_PER_ACTION, VEHICLE_SPEED, WALKING_SPEED
from init_state.wrapper import read_initial_state

# ------------ TESTING DATA MANUALLY ---------------
filename = "instances/EH_W10"
state = read_initial_state(filename)
policy = policies.GreedyPolicy()
state.set_vehicles([policy])


#source = sim.Station(0,capacity=DEFAULT_DEPOT_CAPACITY)
#station1 = sim.Station(1)
#station2 = sim.Station(2)
#station3 = sim.Station(3)
#vehicle1 = sim.Vehicle(1, source, policies.GreedyPolicy(), 0, 6)

class MILP_data():
        def __init__(self):
                #Sets
                
                self.stations = state.stations #{staton_ID: station_object}
                self.stations_with_source_sink = dict() #{staton_ID: station_object}
                self.neighboring_stations = dict()      #{station_ID: [list of station_IDs]}
                self.vehicles = state.vehicles 
                self.time_periods = [0,1,2,3,4,5]
                self.possible_previous_stations_driving = dict()        #{(station_ID,time): [list of station_IDs]}
                self.possible_previous_stations_cycling = dict()        #{(station_ID,time): [list of station_IDs]}
                self.possible_previous_stations_walking = dict()        #{(station_ID,time): [list of station_IDs]}

                #Parameters
                self.W_D = 0.1
                self.W_C = 0.3
                self.W_S = 0.3
                self.W_R = 0.3

                self.neighboring_limit= 0.3 #km

                self.T_D = dict()       #{(station_ID,station_ID):time}
                self.T_DD = dict()       #{(station_ID,station_ID):timeperiods}
                self.T_W = dict()       #{(station_ID,station_ID):time}
                self.T_DW = dict()       #{(station_ID,station_ID):timeperiods}
                self.T_C = dict()       #{(station_ID,station_ID):time}
                self.T_DC = dict()      #{(station_ID,station_ID):timeperiods}
                
                self.T_H = MINUTES_PER_ACTION

                self.L_0 = dict()    #{station_ID:inventory}   
                self.L_T = dict()       #{station_ID:target inventory}

                self.D = dict()         #{(station_ID,timeperiod):demand}

                self.Q_0 = dict()       #{vehicle: inventory}
                self.Q_V = dict()       #{vehicle: capacity}
                self.Q_S = dict()       #{station_ID: capacity}

                self.TAU = 5    #length of time period

                self.DEPOT_ID = 10000

        # ------------ DEFINING MEMBER FUNCTIONS ---------------

        def initialize_stations_with_source_sink(self):
                for station_ID in self.stations:
                        self.stations_with_source_sink[station_ID] = self.stations[station_ID]
                source = sim.Depot(self.DEPOT_ID)     
                self.stations_with_source_sink[source.id] = source

        def set_possible_previous_stations(self, travel_time_dict, possible_previous_stations_dict):
                for j in self.stations:
                        for t in self.time_periods:
                                possible_previous_stations_dict[(j,t)]=[]
                                for i in self.stations_with_source_sink:
                                        if(t-travel_time_dict.get((i,j)) >= 0):
                                                possible_previous_stations_dict[(j,t)].append(i)

        def set_neighboring_stations(self):
                for station in self.stations:
                        self.neighboring_stations[station]=[]
                        for candidate in self.stations:
                                if station != candidate:
                                        distance = self.stations[station].distance_to(self.stations[candidate].get_lat(),self.stations[candidate].get_lon()) 
                                        if distance <= self.neighboring_limit:
                                                self.neighboring_stations[station].append(candidate)


        def initialize_traveltime_dict(self, travel_time_dict, speed, discrete=False, driving=False): 
                for start_station in self.stations_with_source_sink:
                        for end_station in self.stations_with_source_sink:
                                travel_time = (self.stations_with_source_sink[start_station].distance_to(self.stations_with_source_sink[end_station].get_lat()
                                ,self.stations_with_source_sink[end_station].get_lon()) / speed)
                                if driving == True and start_station != end_station:
                                        travel_time += MINUTES_PER_ACTION #parking time included in driving time 
                                if discrete == False:
                                        travel_time_dict[(start_station,end_station)] = travel_time
                                else:
                                        travel_time_dict[(start_station,end_station)] = (travel_time//self.TAU)+1

                for station in self.stations_with_source_sink:
                        if discrete == False:
                                travel_time_dict[(station,self.DEPOT_ID)] = 0 
                                travel_time_dict[(self.DEPOT_ID,station)] = 0
                        else:
                                travel_time_dict[(station,self.DEPOT_ID)] = 1                
                                travel_time_dict[(self.DEPOT_ID,station)] = 1
                
        def set_L_O(self):
                for station in self.stations:
                        self.L_0[station] = self.stations[station].number_of_bikes()

        def set_L_T(self):
                for station in self.stations:
                        #self.L_T[station] = station.get_target_state()                 #must be fixed before simulation
                        self.L_T[station] = self.stations[station].capacity//2

        def set_D(self):
                for station in self.stations:
                        for time in self.time_periods:
                                self.D[(station, time)] = self.TAU*(self.stations[station].get_arrive_intensity(random.randint(0,6), random.randint(8,22)) - self.stations[station].get_leave_intensity(random.randint(0,6), random.randint(8,22))) #must be changed to simulate the real time

        def set_Q_0(self):
                for vehicle in self.vehicles:
                        self.Q_0[vehicle] = len(vehicle.get_bike_inventory())
        
        def set_Q_V(self):
                for vehicle in self.vehicles:
                        self.Q_V[vehicle] = vehicle.bike_inventory_capacity
        
        def set_Q_S(self): 
                for station in self.stations:
                        self.Q_S[station] = self.stations[station].capacity

        
        def initalize_parameters(self):
                self.initialize_stations_with_source_sink()
                self.initialize_traveltime_dict(self.T_D, VEHICLE_SPEED, discrete=False, driving=True)
                self.initialize_traveltime_dict(self.T_DD, VEHICLE_SPEED, discrete=True, driving=True)
                self.initialize_traveltime_dict(self.T_W, WALKING_SPEED, discrete=False, driving=False)
                self.initialize_traveltime_dict(self.T_DW, WALKING_SPEED, discrete=True, driving=False)
                self.initialize_traveltime_dict(self.T_C, BIKE_SPEED, discrete=False, driving=False)
                self.initialize_traveltime_dict(self.T_DC, BIKE_SPEED, discrete=True, driving=False)

                self.set_possible_previous_stations(self.T_DD, self.possible_previous_stations_driving)
                self.set_possible_previous_stations(self.T_DC, self.possible_previous_stations_cycling)
                self.set_possible_previous_stations(self.T_DW, self.possible_previous_stations_walking)

                self.set_neighboring_stations()

                self.set_L_O()
                self.set_L_T()
                
                self.set_D()
                
                self.set_Q_0()
                self.set_Q_V()
                self.set_Q_S()



# d=MILP_data()
# d.initalize_parameters()
# print("heiiiiiiii")    