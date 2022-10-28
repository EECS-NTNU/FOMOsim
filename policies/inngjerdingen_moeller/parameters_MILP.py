
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
from settings import DEFAULT_DEPOT_CAPACITY

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
        self.neighboring_stations = [[] for i in range(len(self.stations_with_source_sink))]
        self.vehicles = [vehicle1]
        self.time_periods = [0,1,2,3,4,5]


        self.possible_previous_stations = [[[],[],[],[],[],[]],[[],[],[],[],[],[]], [[],[],[],[],[],[]], [[],[],[],[],[],[]]] #[station][time_period]

       
            #Parameters
        self.W_D = 0.1
        self.W_C = 0.3
        self.W_S = 0.3
        self.W_R = 0.3

        self.T_D = [[0,0,0,0],
            [0,0,1,6.7],
            [0,1,0,3.1], 
            [0,6.7,3.1,0]]

        self.T_DD = [[1,0,0,0], #number of timeperiods
                [0,1,1,2], 
                [0,1,1,2], 
                [0,2,1,1]]

        self.T_W = [[0,0,0,0],
                [0,0,2,14],
                [0,2,0,8], 
                [0,14,8,0]]

        self.T_DW = [[0,0,0,0],
                [0,0,1,3],
                [0,1,0,2], 
                [0,3,2,0]]

        self.T_C = [[0,0,0,0],
                [0,0,1,10],
                [0,1,0,6], 
                [0,10,6,0]]

        self.T_DC = [[0,0,0,0],
                [0,0,1,2],
                [0,1,0,2], 
                [0,2,2,0]]

        self.T_H = 0.5

        self.L_0 = [0,15,2,1]
        self.L_T = [10,10,10,10,10]

        self.D = [[0,0,0,0,0,0], [0,0.5,0.6,0.2,-0.2,-0.1], [0,-0.5,0.6,-0.2,0,-0.2], [0,0.1,0.3,0.1,-0.3,-0.1]]

        self.Q_0 = {vehicle1: 3}
        self.Q_V = {vehicle1: 6}
        self.Q_S = [0,20,20,20]

        self.tau = 5

        for j in self.stations:
                for t in self.time_periods:
                        for i in self.stations_with_source_sink:
                                if(t-self.T_DD[i][j]>=0):
                                        self.possible_previous_stations[j][t].append(i)

        #self.possible_previous_stations = [[[0], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], [[0], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], [[0], [0, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]
    

d=MILP_data()
print(d.possible_previous_stations)    