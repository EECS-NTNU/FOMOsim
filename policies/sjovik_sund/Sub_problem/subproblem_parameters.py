import math
from settings import VEHICLE_SPEED, MINUTES_CONSTANT_PER_ACTION
 
class MILP_parameters:
 
    def __init__(self, simul, time_horizon = 25, weights = None, tau=5):
        """
        Initialize subproblem parameters for DSBRP (Dynamic Stochastic Bike Rebalancing Problem)
        """
 
        # Handle both Simulator and State objects
        if hasattr(simul, 'state'):
            # simul is a Simulator object
            self.simul = simul
            self.state = simul.state
        else:
            # simul is a State object (called from policy)
            self.state = simul
            self.simul = None  # No simulator reference
 
        # Sets
        self.T = time_horizon                   # Time horizon (number of periods)
        self.tau = tau                          # Length of each time period (in minutes)
        self.N = []                             # Set of station IDs (list of integers/strings)
        self.s = None                           # Source node (depot start)
        self.d = None                           # Sink node (depot end)
        self.V = []                             # Set of vehicle IDs (list)
       
        # Objective function weights (must match model expectations: w_S, w_C, w_D)
        if weights == None:  # default
            self.w_S = 0.45  # Weight for starvation penalty
            self.w_C = 0.45  # Weight for congestion penalty
            self.w_D = 0.1   # Weight for deviation from target state penalty
            self.r_M = 0.45   # No maintenance reward by default
        else:
            self.w_S = weights[0]  # Starvation weight
            self.w_C = weights[1]  # Congestion weight
            self.w_D = weights[2]  # Deviation weight
            self.r_M = weights[3]  # Maintenance reward weight
       
        # Travel time parameters
        self.T_D = {}          # Travel time from i to j in minutes (dict with keys (i,j))
        self.T_DD = {}         # Discretized travel time periods from i to j (dict with keys (i,j))
        self.T_L = 0.5         # Loading/unloading time per bike in minutes (float)
       
        # Maintenance time parameters
        self.T_M_min = {}      # Minimum maintenance time at station i in minutes (dict with keys i)
        self.T_M_max = {}      # Maximum maintenance time at station i in minutes (dict with keys i)
       
        # Vehicle capacity parameters
        self.Q_V = {}          # Capacity of vehicle v (dict with keys v)
        self.Q_V0 = {}         # Initial load of vehicle v (dict with keys v)
       
        # Station inventory parameters
        self.I_N = {}          # Capacity of station i (dict with keys i)
        self.I_N0 = {}         # Initial inventory at station i (dict with keys i)
        self.I_T = {}          # Target inventory at station i at horizon T (dict with keys i)
       
        # Demand parameters
        self.D = {}            # Net demand at station i in period t (dict with keys (i,t))
                               # Negative values = more pickups than returns
       
   
    def to_dict(self):
        """
        Convert parameters to dictionary format for Gurobi model input
        """
        return {
            "T": self.T,
            "tau": self.tau,
            "N": self.N,
            "s": self.s,
            "d": self.d,
            "V": self.V,
            "T_D": self.T_D,
            "T_DD": self.T_DD,
            "T_L": self.T_L,
            "T_M_min": self.T_M_min,
            "T_M_max": self.T_M_max,
            "Q_V": self.Q_V,
            "Q_V0": self.Q_V0,
            "I_N": self.I_N,
            "I_N0": self.I_N0,
            "I_T": self.I_T,
            "D": self.D,
            "w_S": self.w_S,
            "w_C": self.w_C,
            "w_D": self.w_D,
            "r_M": self.r_M
        }
   
    def print_all_params(self):
        """
        Print all parameters for debugging purposes
        """
        print("=" * 80)
        print("DSBRP SUBPROBLEM PARAMETERS")
        print("=" * 80)
       
        print("\n--- TIME PARAMETERS ---")
        print(f"Time horizon (T): {self.T}")
        print(f"Period length (tau): {self.tau}")
       
        print("\n--- SETS ---")
        print(f"Stations (N): {self.N}")
        print(f"Source node (s): {self.s}")
        print(f"Sink node (d): {self.d}")
        print(f"Vehicles (V): {self.V}")
       
        print("\n--- TRAVEL TIME PARAMETERS ---")
        print(f"Travel times (T_D): {len(self.T_D)} entries")
        if self.T_D:
            for key, val in list(self.T_D.items())[:5]:
                print(f"  {key}: {val}")
            if len(self.T_D) > 5:
                print(f"  ... and {len(self.T_D) - 5} more")
       
        print(f"Discretized travel periods (T_DD): {len(self.T_DD)} entries")
        if self.T_DD:
            for key, val in list(self.T_DD.items())[:5]:
                print(f"  {key}: {val}")
            if len(self.T_DD) > 5:
                print(f"  ... and {len(self.T_DD) - 5} more")
       
        print(f"Loading/unloading time per bike (T_L): {self.T_L}")
       
        print("\n--- MAINTENANCE PARAMETERS ---")
        print(f"Minimum maintenance times (T_M_min): {self.T_M_min}")
        print(f"Maximum maintenance times (T_M_max): {self.T_M_max}")
       
        print("\n--- VEHICLE PARAMETERS ---")
        print(f"Vehicle capacities (Q_V): {self.Q_V}")
        print(f"Initial vehicle loads (Q_V0): {self.Q_V0}")
       
        print("\n--- STATION INVENTORY PARAMETERS ---")
        print(f"Station capacities (I_N): {self.I_N}")
        print(f"Initial inventories (I_N0): {self.I_N0}")
        print(f"Target inventories (I_T): {self.I_T}")
       
        print("\n--- DEMAND PARAMETERS ---")
        print(f"Demand (D): {len(self.D)} entries")
        if self.D:
            for key, val in list(self.D.items())[:5]:
                print(f"  Station {key[0]}, Period {key[1]}: {val}")
            if len(self.D) > 5:
                print(f"  ... and {len(self.D) - 5} more")
       
        print("\n--- OBJECTIVE FUNCTION WEIGHTS ---")
        print(f"Starvation weight (w_S): {self.w_S}")
        print(f"Congestion weight (w_C): {self.w_C}")
        print(f"Deviation weight (w_D): {self.w_D}")
        print(f"Maintenance reward (r_M): {self.r_M}")
       
        print("=" * 80)
   
    def initalize_parameters(self):
        """
        Initialize all parameters from the simulation state.
        This is the main method that populates all the data structures.
        """
        # Initialize sets
        self._initialize_stations()
        self._initialize_vehicles()
        self._initialize_depot()
       
        # Initialize travel time parameters
        self._initialize_travel_times()
       
        # Initialize station parameters
        self._initialize_station_inventories()
        self._initialize_station_capacities()
        self._initialize_target_inventories()
       
        # Initialize vehicle parameters
        self._initialize_vehicle_capacities()
        self._initialize_vehicle_loads()
       
        # Initialize demand
        self._initialize_demand()
       
        # Initialize maintenance parameters (set to 0 if not using maintenance)
        self._initialize_maintenance()
   
    def _initialize_stations(self):
        """Extract station IDs from the simulation state."""
        # Get original station IDs (strings like 'S0', 'S1', etc.)
        original_station_ids = [station.id for station in self.state.stations.values()]
       
        # Create integer indices for MILP (0, 1, 2, ..., n-1)
        self.N = list(range(len(original_station_ids)))
       
        # Create mappings between original IDs and MILP indices
        self.station_id_to_index = {sid: i for i, sid in enumerate(original_station_ids)}
        self.index_to_station_id = {i: sid for i, sid in enumerate(original_station_ids)}
   
    def _initialize_vehicles(self):
        """Extract vehicle IDs from the simulation state."""
        # Get original vehicle IDs
        original_vehicle_ids = [vehicle.id for vehicle in self.state.get_vehicles()]
       
        # Create integer indices for MILP (0, 1, 2, ..., m-1)
        self.V = list(range(len(original_vehicle_ids)))
       
        # Create mappings
        self.vehicle_id_to_index = {vid: i for i, vid in enumerate(original_vehicle_ids)}
        self.index_to_vehicle_id = {i: vid for i, vid in enumerate(original_vehicle_ids)}
   
    def _initialize_depot(self):
        """
        Set source and sink nodes (depot).
        Use integers that don't conflict with station indices.
        """
        # Use negative integers for depot nodes (stations use 0, 1, 2, ...)
        self.s = -1  # Source node (depot/start)
        self.d = -2  # Sink node (depot/end)
   
    def _initialize_travel_times(self):
        """
        Calculate travel times between all station pairs.
        T_D: actual travel time in minutes
        T_DD: discretized travel time in periods
        """
        stations = self.state.stations
       
        # Calculate travel times between all station pairs using MILP indices
        for i_idx in self.N:
            for j_idx in self.N:
                if i_idx == j_idx:
                    # Same station - minimal time
                    self.T_D[(i_idx, j_idx)] = 0.0
                    self.T_DD[(i_idx, j_idx)] = 1
                else:
                    # Get actual station objects using mapping
                    station_i_id = self.index_to_station_id[i_idx]
                    station_j_id = self.index_to_station_id[j_idx]
                    station_i = stations[station_i_id]
                    station_j = stations[station_j_id]
                   
                    # Distance in km
                    distance = station_i.distance_to(station_j.get_lat(), station_j.get_lon())
                   
                    # Travel time in minutes = (distance / speed) * 60 + constant time
                    travel_time = (distance / VEHICLE_SPEED) * 60 + MINUTES_CONSTANT_PER_ACTION
                   
                    self.T_D[(i_idx, j_idx)] = travel_time
                   
                    # Discretized travel time = ceil(travel_time / tau)
                    self.T_DD[(i_idx, j_idx)] = max(1, math.ceil(travel_time / self.tau))
       
        # Add depot connections (0 time to/from depot for simplicity)
        for i_idx in self.N:
            # From source depot to station
            self.T_D[(self.s, i_idx)] = 0.0
            self.T_DD[(self.s, i_idx)] = 1
            # From station to sink depot
            self.T_D[(i_idx, self.d)] = 0.0
            self.T_DD[(i_idx, self.d)] = 1
            # From sink depot to station (for completeness)
            self.T_D[(self.d, i_idx)] = 0.0
            self.T_DD[(self.d, i_idx)] = 1
            # From station to source depot (for completeness)
            self.T_D[(i_idx, self.s)] = 0.0
            self.T_DD[(i_idx, self.s)] = 1
       
        # Depot to itself and depot-to-depot
        self.T_D[(self.s, self.s)] = 0.0
        self.T_D[(self.d, self.d)] = 0.0
        self.T_D[(self.s, self.d)] = 0.0
        self.T_D[(self.d, self.s)] = 0.0
        self.T_DD[(self.s, self.s)] = 1
        self.T_DD[(self.d, self.d)] = 1
        self.T_DD[(self.s, self.d)] = 1
        self.T_DD[(self.d, self.s)] = 1
   
    def _initialize_station_inventories(self):
        """Get current inventory at each station."""
        for station_idx in self.N:
            station_id = self.index_to_station_id[station_idx]
            station = self.state.stations[station_id]
            # Count available bikes at the station
            self.I_N0[station_idx] = station.number_of_bikes()
   
    def _initialize_station_capacities(self):
        """Get capacity of each station."""
        for station_idx in self.N:
            station_id = self.index_to_station_id[station_idx]
            station = self.state.stations[station_id]
            self.I_N[station_idx] = station.capacity
   
    def _initialize_target_inventories(self):
        """Get target inventory for each station at the end of the horizon."""
        for station_idx in self.N:
            station_id = self.index_to_station_id[station_idx]
            station = self.state.stations[station_id]
            # Get target state for current day/hour
            self.I_T[station_idx] = station.get_target_state(self.state.day(), self.state.hour())
   
    def _initialize_vehicle_capacities(self):
        """Get capacity of each vehicle."""
        for vehicle_idx in self.V:
            vehicle_id = self.index_to_vehicle_id[vehicle_idx]
            vehicle = self.state.vehicles[vehicle_id]
            self.Q_V[vehicle_idx] = vehicle.bike_inventory_capacity
   
    def _initialize_vehicle_loads(self):
        """Get current load (number of bikes) on each vehicle."""
        for vehicle_idx in self.V:
            vehicle_id = self.index_to_vehicle_id[vehicle_idx]
            vehicle = self.state.vehicles[vehicle_id]
            self.Q_V0[vehicle_idx] = len(vehicle.get_bike_inventory())
   
    def _initialize_demand(self):
        """
        Initialize demand predictions for each station and time period.
        D[(station_idx, period)] = net demand (arrivals - departures)
       
        For now, using arrival/departure intensities from the demand model.
        Positive value = net arrivals (more bikes coming in)
        Negative value = net departures (more bikes leaving)
        """
        for station_idx in self.N:
            station_id = self.index_to_station_id[station_idx]
            station = self.state.stations[station_id]
            day = self.state.day()
            hour = self.state.hour()
           
            for period in range(1, self.T + 1):
                # Calculate expected demand for this time period
                # tau is in minutes, intensities are per hour
                time_fraction = self.tau / 60.0  # Convert period length to hours
               
                # Get arrival and departure intensities (bikes per hour)
                arrival_intensity = station.get_arrive_intensity(day, hour)
                departure_intensity = station.get_leave_intensity(day, hour)
               
                # Net demand = arrivals - departures for this period
                net_demand = time_fraction * (arrival_intensity - departure_intensity)
               
                self.D[(station_idx, period)] = net_demand
   
    def _initialize_maintenance(self):
        """
        Initialize maintenance parameters.
        Set to 0 for all stations if not using maintenance.
        """
        for station_idx in self.N:
            self.T_M_min[station_idx] = 0  # No minimum maintenance time
            self.T_M_max[station_idx] = 5  # No maximum maintenance time
 
 
 