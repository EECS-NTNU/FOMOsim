
class Station():
    def __init__(self, station_id):
        self.station_ID = station_id
        self.neighboring_stations = []
    
    def add_neighbors(self, neighbors):
        for neighbor in neighbors:
            self.neighboring_stations.append(neighbor)

class Vehicle():
    def __init__(self, vehicle_id):
        self.vehicle_ID = vehicle_id
        self.carrying_capacity = 20     #hard coded for now

# ------------ TESTING DATA MANUALLY ---------------
source = Station(0)
station1 = Station(1)
station2 = Station(2)
station3 = Station(3)

vehicle = Vehicle(1)

class MILP_data():
    def __init__(self):
        #Sets
        #self.stations = [station1, station2, station3]
        self.stations = [1, 2, 3]
        #self.stations_with_source_sink = [source, station1, station2, station3]
        self.stations_with_source_sink = [0, 1, 2, 3]
        self.neighboring_stations = [[],[2],[1],[]]
        #self.vehicles = [vehicle]
        self.vehicles = [1]
        self.time_periods = [0,1,2,3,4,5]


            #Parameters
        self.W_D = 0.1
        self.W_C = 0.3
        self.W_S = 0.3
        self.W_R = 0.3

        self.T_D = [[0,0,0,0],
            [0,0,1,6.7],
            [0,1,0,3.1], 
            [0,6.7,3.1,0]]

        self.T_DD = [[0,0,0,0], #number of timeperiods
                [0,0,1,2], 
                [0,1,0,2], 
                [0,2,1,0]]

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

        self.D = [[0.5,0.6,0.2,-0.2,-0.1], [-0.5,0.6,-0.2,0,-0.2], [0.1,0.3,0.1,-0.3,-0.1]]

        self.Q_0 = [3]
        self.Q_V = [6]
        self.Q_S = [0,20,20,20]

        self.tau = 5
