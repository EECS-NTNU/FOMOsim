
#from policies.gleditsch_hagen.utils import calculate_net_demand


####################################
# set the right working directory #
###################################

import os 
import sys
from pathlib import Path
import json
 
path = Path(__file__).parents[2]        # The path seems to be correct either way, sys.path.insert makes the difference
os.chdir(path)
# print(os. getcwd())
sys.path.insert(0, '') #make sure the modules are found in the new working directory

##############################################################


from settings import BIKE_SPEED, MINUTES_PER_ACTION, VEHICLE_SPEED, WALKING_SPEED, MINUTES_CONSTANT_PER_ACTION

class MILP_data():
        def __init__(self, simul, time_horizon=25, tau=5, weights = None):
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
                if weights == None: #default 
                        self.W_S = 0.45
                        self.W_R = 0.45
                        self.W_D = 0.1
                else: 
                        self.W_S = weights[0] 
                        self.W_R = weights[1]
                        self.W_D = weights[2]
        
                        
                self.W_C = self.W_R #only in use when roaming = False

                self.neighboring_limit= 0.6 #km

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

                self.discounting_factors = []

        # ------------ DEFINING MEMBER FUNCTIONS ---------------

        def initialize_time_periods(self):
                num_time_periods=self.time_horizon//self.TAU  #number of time periods are rounded down to nearest integer
                for period in range(0,int(num_time_periods)+1):
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
                        
                        #Code for creating more variation in initial inventory 
                        # if station%2 == 0: #even number stations:
                        #         self.L_0[station] = min(self.stations[station].number_of_bikes()+5,self.stations[station].capacity) 
                        # else:
                        #         self.L_0[station] = max(self.stations[station].number_of_bikes()-5,0) 

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
                        # self.Q_0[vehicle] = 10
        
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

        def initialize_discounting_factors(self, end_factor): #end_factor is discounting factor in last time period
                nPeriods = len(self.time_periods)-1 #time_periods less period 0 
                rate = (1/end_factor)**(1/nPeriods)-1
                self.discounting_factors.append(0) #0 on element 0, will not be used
                for period in range(0,nPeriods):
                        discount_factor = 1/((1+rate)**period) #1 in period 1
                        self.discounting_factors.append(discount_factor)

        

        def dump_static_data(self): 
                filename = 'policies/inngjerdingen_moeller/saved_time_data/' + (self.state.mapdata[0]).split('.')[0].split('/')[1] +'_static_data.json'
                T_D = {str(k): v for k, v in self.T_D.items()}
                T_DD = {str(k): v for k, v in self.T_DD.items()}
                T_C = {str(k): v for k, v in self.T_C.items()}
                T_DC = {str(k): v for k, v in self.T_DC.items()}
                T_W = {str(k): v for k, v in self.T_W.items()}
                T_DW = {str(k): v for k, v in self.T_DW.items()}
                json_dict = {"T_D": T_D, "T_DD": T_DD, "T_C": T_C, "T_DC": T_DC, "T_W":T_W, "T_DW":T_DW, "neighboring_stations": self.neighboring_stations}
        
                json_object = json.dumps(json_dict,indent=4)
                with open(filename, "w") as outfile:
                        outfile.write(json_object)

        def read_static_data(self):
                filename = 'policies/inngjerdingen_moeller/saved_time_data/' + (self.state.mapdata[0]).split('.')[0].split('/')[1] +'_static_data.json'
                with open(filename,'r') as infile:
                        json_dict = json.load(infile)
                self.convert_dict(json_dict["T_D"], self.T_D)
                self.convert_dict(json_dict["T_DD"], self.T_DD)
                self.convert_dict(json_dict["T_C"], self.T_C)
                self.convert_dict(json_dict["T_DC"], self.T_DC)
                self.convert_dict(json_dict["T_W"], self.T_W)
                self.convert_dict(json_dict["T_DW"], self.T_DW)
                for key, value in json_dict["neighboring_stations"].items():
                        self.neighboring_stations[int(key)] = value
                
        def convert_dict(self, string_dict, original_dict):
                for key, value in string_dict.items():
                        tuple_key = tuple(int(num) for num in key.replace('(', '').replace(')', '').replace('...', '').split(', '))
                        original_dict[tuple_key] = value


        def initalize_parameters(self):
                self.initialize_time_periods()
                self.initialize_stations_with_source_sink()
                self.initialize_vehicles()
                if os.path.exists('policies/inngjerdingen_moeller/saved_time_data/' + (self.state.mapdata[0]).split('.')[0].split('/')[1] +'_static_data.json'):
                        self.read_static_data()
                else:
                        self.initialize_traveltime_dict(self.T_D, VEHICLE_SPEED, discrete=False, driving=True)
                        self.initialize_traveltime_dict(self.T_DD, VEHICLE_SPEED, discrete=True, driving=True)
                        self.initialize_traveltime_dict(self.T_W, WALKING_SPEED, discrete=False, driving=False)
                        self.initialize_traveltime_dict(self.T_DW, WALKING_SPEED, discrete=True, driving=False)
                        self.initialize_traveltime_dict(self.T_C, BIKE_SPEED, discrete=False, driving=False)
                        self.initialize_traveltime_dict(self.T_DC, BIKE_SPEED, discrete=True, driving=False)
                        self.set_neighboring_stations()
                        self.dump_static_data()
                
                self.initialize_vehicle_ETAs()
                self.initialize_discounting_factors(0.9)

                self.set_possible_previous_stations(self.T_DD, self.possible_previous_stations_driving, driving=True)
                self.set_possible_previous_stations(self.T_DC, self.possible_previous_stations_cycling)
                self.set_possible_previous_stations(self.T_DW, self.possible_previous_stations_walking, walking= True)

                self.set_L_O()
                self.set_L_T()

                self.set_D(self.simul.day(), self.simul.hour())    

                self.set_Q_0()
                self.set_Q_V()
                self.set_Q_S()

        def print_neighbor_info(self, station_ID):
                for neighbor in self.neighboring_stations[station_ID]:
                        print("Station ID:", str(neighbor), "Inventory:", str(self.L_0[neighbor]), "Capacity:", str(self.Q_S[neighbor]), "Walking distance:", str(round(self.T_D[(neighbor, station_ID)],1)))

        def deep_dive_test_2(self):

                # station_list = [36,51,48,19,12,9]
                # for station in station_list:
                #         self.L_0[station] = 0
                #         # self.L_T[station] = 10
                #         for timeperiod in self.time_periods:
                #                 self.D[(station, timeperiod)] = -0.5
                


                # Example solution:
                self.L_0[35] =15
                self.L_0[54] = 15
                self.L_0[40] = 0
                self.L_0[27] = 0
                
                for timeperiod in self.time_periods:
                        self.D[(35, timeperiod)] = 1
                        self.D[(54, timeperiod)] = 1
                        self.D[(40, timeperiod)] = -1
                        self.D[(27, timeperiod)] = -1