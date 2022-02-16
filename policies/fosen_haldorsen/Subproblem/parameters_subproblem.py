
class ParameterSub:

    def __init__(self, route, vehicle, pattern, customer_arrivals, L_CS, L_FS, base_violations, V_0, D_O, base_deviations,
                 weights, hour):
        # Sets
        self.stations = [i for i in range(len(route.stations))]
        self.charging_stations = list()
        self.non_charging_stations = list()
        self.depot_index = None
        for i in range(1, len(route.stations)):  # Don't include start station in subsets
            if route.stations[i].charging_station:
                self.charging_stations.append(i)
            else:
                self.non_charging_stations.append(i)
            if route.stations[i].depot:
                self.depot_index = i
        self.stations += [len(route.stations)]

        # Pattern
        # Q_B, Q_CCL, Q_FCL, Q_CCU, Q_FCU
        self.Q_B = pattern[0]
        self.Q_CCL = pattern[1]
        self.Q_FCL = pattern[2]
        self.Q_CCU = pattern[3]
        self.Q_FCU = pattern[4]

        # Station specific
        self.Q_S = [station.station_cap for station in route.stations] + [0]
        self.L_CS = L_CS + [0]
        self.L_FS = L_FS + [0]
        self.I_IC = [customer_arrivals[i][0] for i in range(len(customer_arrivals))] + [0]
        self.I_IF = [customer_arrivals[i][1] for i in range(len(customer_arrivals))] + [0]
        self.I_OC = [customer_arrivals[i][2] for i in range(len(customer_arrivals))] + [0]
        self.O = [station.get_ideal_state(hour) for station in route.stations] + [0]

        # Vehicle specific
        self.Q_BV = vehicle.battery_capacity
        self.Q_CV = vehicle.bike_capacity + self.Q_CCL + self.Q_FCL - max(0, self.Q_CCU + self.Q_FCU)
        if route.stations[0].depot:
            self.depot_index = 0
            self.L_BV = vehicle.battery_capacity
        else:
            self.L_BV = vehicle.current_batteries - self.Q_B
        self.L_CV = vehicle.current_charged_bikes + self.Q_CCL - self.Q_CCU
        self.L_FV = vehicle.current_flat_bikes + self.Q_FCL - self.Q_FCU

        # Base Violations
        self.V = base_violations + [0]
        self.V_O = V_0
        self.R_O = 0
        self.D = base_deviations + [0]
        self.D_O = D_O
        if route.stations[0].charging_station:
            self.R_O = max(0, self.Q_FCU - self.Q_FCL)

        # Weights
        self.W_V, self.W_R, self.W_D, self.W_N, self.W_L = weights

        # self.print_all_params(pattern)

    def print_all_params(self, pattern):
        print("Stations: ", self.stations)
        print("Charging Stations: ", self.charging_stations)
        print("Non Charging Stations: ", self.non_charging_stations)
        print("Depot index: ", self.depot_index)

        print("Pattern: ", pattern)

        print("Q_S: ", self.Q_S)
        print("Ideal state: ", self.O)
        print("L_CS: ", self.L_CS)
        print("L_FS: ", self.L_FS)
        print("I_IC: ", self.I_IC)
        print("I_IF: ", self.I_IF)
        print("I_OC: ", self.I_OC)

        print("Q_BV: ", self.Q_BV)
        print("Q_CV: ", self.Q_CV)
        print("L_BV: ", self.L_BV)
        print("L_CV: ", self.L_CV)
        print("L_FV: ", self.L_FV)

        print("Base Violations: ", self.V)
        print("V_O: ", self.V_O)
        print("Base Deviations: ", self.D)
        print("D_O: ", self.D_O)
        print("R_O: ", self.R_O)
