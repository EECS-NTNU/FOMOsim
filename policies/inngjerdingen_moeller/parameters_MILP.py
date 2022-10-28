
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


import sim
import policies
from settings import DEFAULT_DEPOT_CAPACITY, MINUTES_PER_ACTION

# ------------ TESTING DATA MANUALLY ---------------
source = sim.Station(0,capacity=DEFAULT_DEPOT_CAPACITY)
station1 = sim.Station(1)
station2 = sim.Station(2)
station3 = sim.Station(3)

vehicle1 = sim.Vehicle(1, source, policies.GreedyPolicy(), 0, 6)

class MILP_data():
        def __init__(self):
                #Sets
                # self.stations = [1, 2, 3]
                self.stations = [station1, station2, station3]
                # self.stations_with_source_sink = [0, 1, 2, 3]
                self.stations_with_source_sink = [source, station1, station2, station3]
                self.neighboring_stations = dict()      #{station: [list of stations]}
                self.vehicles = [vehicle1]
                self.time_periods = [0,1,2,3,4,5]
                self.possible_previous_stations_driving = dict()        #{(station,time): [list of stations]}
                self.possible_previous_stations_cycling = dict()        #{(station,time): [list of stations]}

                #Parameters
                self.W_D = 0.1
                self.W_C = 0.3
                self.W_S = 0.3
                self.W_R = 0.3

                self.T_D = dict()       #{(station,station):time}
                self.T_DD = dict()       #{(station,station):timeperiods}
                self.T_W = dict()       #{(station,station):time}
                self.T_DW = dict()       #{(station,station):timeperiods}
                self.T_C = dict()       #{(station,station):time}
                self.T_DC = dict()      #{(station,station):timeperiods}
                
                self.T_H = MINUTES_PER_ACTION

                self.L_0 = dict()    #{station:inventory}   
                self.L_T = dict()       #{station:target inventory}

                self.D = dict()         #{(station,timeperiod):demand}

                self.Q_0 = dict()       #{vehicle: inventory}
                self.Q_V = dict()       #{vehicle: capacity}
                self.Q_S = dict()       #{station:capacity}

                self.tau = 5    #length of time period

        # ------------ DEFINING MEMBER FUNCTIONS ---------------

        def set_possible_previous_stations_driving(self):
                for j in self.stations:
                        for t in self.time_periods:
                                for i in self.stations_with_source_sink:
                                        if(t-self.T_DD[i][j]>=0):
                                                self.possible_previous_stations_driving[(j,t)] = i

        def set_possible_previous_stations_cycling(self):
                for j in self.stations:
                        for t in self.time_periods:
                                for i in self.stations_with_source_sink:
                                        if(t-self.T_DC[i][j]>=0):
                                                self.possible_previous_stations_cycling[(j,t)] = i
    

d=MILP_data()
d.set_possible_previous_stations()
print(d.possible_previous_stations)    