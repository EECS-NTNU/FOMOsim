
#from policies.gleditsch_hagen.utils import calculate_net_demand


####################################
# set the right working directory #
###################################

import os 
import sys
from pathlib import Path

path = Path(__file__).parents[2]        # The path seems to be correct either way, sys.path.insert makes the difference
os.chdir(path)
# print(os. getcwd())
sys.path.insert(0, '') #make sure the modules are found in the new working directory

##############################################################


from settings import BIKE_SPEED, MINUTES_PER_ACTION, VEHICLE_SPEED, WALKING_SPEED, MINUTES_CONSTANT_PER_ACTION

class MILP_data():
        def __init__(self, simul, time_horizon=25, tau=5):
                #Sets
                
                self.simul = simul
                self.state = simul.state
                self.stations = self.state.stations #{staton_ID: station_object}
                self.stations_with_source_sink = dict() #{staton_ID: station_object}
                self.neighboring_stations = dict()      #{station_ID: [list of station_IDs]}
                self.vehicles = dict()                  #{vehicle_ID: vehicle_object}
                self.time_periods = []
                self.possible_previous_stations_driving = dict()        #{(station_ID,time): [list of station_IDs]}
                self.possible_previous_stations_cycling = dict()        #{(station_ID,time): [list of station_IDs]}
                self.possible_previous_stations_walking = dict()        #{(station_ID,time): [list of station_IDs]}

                #Parameters
                self.W_D = 0.1
                self.W_C = 0.45
                self.W_S = 0.45
                self.W_R = 0.45

                self.neighboring_limit= 0.35 #km

                self.TW_max = (self.neighboring_limit/WALKING_SPEED)*60      #max walking time between neighbors in minutes
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

                self.TAU = tau    #length of time period, minutes
                self.time_horizon = time_horizon #length of time horizon in minutes

                self.DEPOT_ID = -1

        # ------------ DEFINING MEMBER FUNCTIONS ---------------

        def initialize_time_periods(self):
                num_time_periods=self.time_horizon//self.TAU  #number of time periods are rounded down to nearest integer
                for period in range(0,num_time_periods+1):
                        self.time_periods.append(period)
        

        def initialize_stations_with_source_sink(self):
                for station_ID in self.stations:
                        self.stations_with_source_sink[station_ID] = self.stations[station_ID]
                source = self.DEPOT_ID     
                self.stations_with_source_sink[source] = source

        def set_possible_previous_stations(self, travel_time_dict, possible_previous_stations_dict, driving=False, walking=False):
                for j in self.stations:
                        for t in self.time_periods:
                                possible_previous_stations_dict[(j,t)]=[]
                                if driving == True:
                                        for i in self.stations_with_source_sink:
                                                if(t-travel_time_dict.get((i,j)) >= 0):
                                                        possible_previous_stations_dict[(j,t)].append(i)
                                else:
                                        for i in self.stations:
                                                if i != j:
                                                        if(t-travel_time_dict.get((i,j)) >= 1):
                                                                if walking == True and i not in self.neighboring_stations[j]:
                                                                        pass 
                                                                else:
                                                                        possible_previous_stations_dict[(j,t)].append(i)

        def set_neighboring_stations(self):
                for station in self.stations:
                        self.neighboring_stations[station]=[]
                        for candidate in self.stations:
                                if station != candidate:
                                        distance = self.stations[station].distance_to(self.stations[candidate].get_lat(),self.stations[candidate].get_lon()) 
                                        # print(station,",",candidate,": ",distance)
                                        if distance <= self.neighboring_limit:
                                                self.neighboring_stations[station].append(candidate)

        def initialize_vehicles(self):
                for vehicle in self.state.vehicles:
                        self.vehicles[vehicle.vehicle_id] = vehicle


        def initialize_traveltime_dict(self, travel_time_dict, speed, discrete=False, driving=False): 
                for start_station in self.stations:
                        for end_station in self.stations:
                                travel_time = (self.stations[start_station].distance_to(self.stations[end_station].get_lat()
                                ,self.stations_with_source_sink[end_station].get_lon()) / speed) * 60     # minutes
                                if driving == True and start_station != end_station:
                                        travel_time += MINUTES_CONSTANT_PER_ACTION #parking time included in driving time (1 min)
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
                        self.L_T[station] = self.stations[station].get_target_state(self.simul.day(), self.simul.hour())
                        #self.L_T[station] = (self.stations[station].capacity//2)


        def set_D(self, day, hour):
                for station in self.stations:
                        for time in self.time_periods:
                                self.D[(station, time)] = (self.TAU/60)*(self.stations[station].get_arrive_intensity(day, hour) - self.stations[station].get_leave_intensity(day, hour)) # demand per time period

        def set_Q_0(self):
                for vehicle in self.vehicles: 
                        self.Q_0[vehicle] = len(self.vehicles[vehicle].get_bike_inventory())
        
        def set_Q_V(self):
                for vehicle in self.vehicles:
                        self.Q_V[vehicle] = self.vehicles[vehicle].bike_inventory_capacity
        
        def set_Q_S(self): 
                for station in self.stations:
                        self.Q_S[station] = self.stations[station].capacity

        def initialize_vehicle_ETAs(self):
                for vehicle in self.vehicles:
                        if self.vehicles[vehicle].eta > 0:
                                self.T_D[(-1, self.vehicles[vehicle].location.id)] = (self.vehicles[vehicle].eta - self.simul.time)
                                self.T_DD[(-1, self.vehicles[vehicle].location.id)] = ((self.vehicles[vehicle].eta - self.simul.time)//self.TAU)+1


        
        def initalize_parameters(self):
                self.initialize_time_periods()
                self.initialize_stations_with_source_sink()
                self.initialize_vehicles()
                self.initialize_traveltime_dict(self.T_D, VEHICLE_SPEED, discrete=False, driving=True)
                self.initialize_traveltime_dict(self.T_DD, VEHICLE_SPEED, discrete=True, driving=True)
                self.initialize_traveltime_dict(self.T_W, WALKING_SPEED, discrete=False, driving=False)
                self.initialize_traveltime_dict(self.T_DW, WALKING_SPEED, discrete=True, driving=False)
                self.initialize_traveltime_dict(self.T_C, BIKE_SPEED, discrete=False, driving=False)
                self.initialize_traveltime_dict(self.T_DC, BIKE_SPEED, discrete=True, driving=False)
                self.initialize_vehicle_ETAs()
                
                self.set_neighboring_stations()
                
                self.set_possible_previous_stations(self.T_DD, self.possible_previous_stations_driving, driving=True)
                self.set_possible_previous_stations(self.T_DC, self.possible_previous_stations_cycling)
                self.set_possible_previous_stations(self.T_DW, self.possible_previous_stations_walking, walking= True)

                self.set_L_O()
                self.set_L_T()
                
                self.set_D(self.simul.day(), self.simul.hour())    
                
                self.set_Q_0()
                self.set_Q_V()
                self.set_Q_S()
