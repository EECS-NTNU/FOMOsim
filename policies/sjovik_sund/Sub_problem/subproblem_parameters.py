import math
from settings import VEHICLE_SPEED, MINUTES_CONSTANT_PER_ACTION

# Maintenance constants - shared across MILP and policy
TIME_PER_BIKE_MAINTENANCE = 3.0  # Minutes to service one bike (matches MINUTES_PER_ACTION)
MAX_BIKES_PER_VISIT = 10  # Maximum bikes that can be serviced at one station visit
 
class MILP_parameters:
 
    def __init__(self, simul, time_horizon = 6, weights = None, tau=5):
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
        self.stations = []                             # Set of station IDs (list of integers/strings)
        self.source = None                           # Source node (depot start)
        self.sink = None                           # Sink node (depot end)
        self.V = []                             # Set of vehicle IDs (list)
       
        # Objective function weights (must match model expectations: w_S, w_C, w_D)
        if weights == None:  # default
            self.w_S = 0.45  # Weight for starvation penalty
            self.w_C = 0.45  # Weight for congestion penalty
            self.w_D = 0.1   # Weight for deviation from target state penalty
            self.r_M = 0.01   # No maintenance reward by default
        else:
            self.w_S = weights[0]  # Starvation weight
            self.w_C = weights[1]  # Congestion weight
            self.w_D = weights[2]  # Deviation weight
            self.r_M = weights[3]  # Maintenance reward weight
       
        # Travel time parameters
        self.T_D = {}          # {(station_idx_i, station_idx_j): travel_time_minutes} - Travel time from i to j in minutes
        self.T_DD = {}         # {(station_idx_i, station_idx_j): num_periods} - Discretized travel time periods from i to j
        self.T_L = 3         # Loading/unloading time per bike in minutes (float)
       
        # Maintenance time parameters
        self.T_M_min = {}      # {station_idx: min_time_minutes} - Minimum maintenance time at station i
        self.T_M_max = {}      # {station_idx: max_time_minutes} - Maximum maintenance time at station i
       
        # Vehicle capacity parameters
        self.Q_V = {}          # {vehicle_idx: capacity} - Capacity of vehicle v (max bikes it can carry)
        self.Q_V0 = {}         # {vehicle_idx: initial_load} - Initial load of vehicle v (bikes currently on vehicle)
        self.eta = {}          # {vehicle_idx: station_idx} - Initial destination station for vehicle v (where it's headed at t=0)
       
        # Station inventory parameters
        self.Q_S = {}          # {station_idx: capacity} - Capacity of station i (max bikes station can hold)
        self.I_N0 = {}         # {station_idx: initial_inventory} - Initial inventory at station i (bikes currently at station)
        self.I_T = {}          # {station_idx: target_inventory} - Target inventory at station i at horizon T
       
        # Demand parameters
        self.D = {}            # {(station_idx, period): net_demand} - Net demand at station i in period t
                               # Positive = net arrivals (more bikes coming in), Negative = net departures (more bikes leaving)
       
 
   
    def _initialize_stations(self):
        # Get original station IDs (strings like 'S0', 'S1', etc.)
        original_station_ids = [station.id for station in self.state.stations.values()]
       
        # Create integer indices for MILP (0, 1, 2, ..., n-1)
        self.stations = list(range(len(original_station_ids)))
       
        # Create mappings between original IDs and MILP indices
        self.station_id_to_index = {sid: i for i, sid in enumerate(original_station_ids)}
        self.index_to_station_id = {i: sid for i, sid in enumerate(original_station_ids)}
   
    def _initialize_vehicles(self):
        # Get original vehicle IDs
        original_vehicle_ids = [vehicle.id for vehicle in self.state.get_vehicles()]
       
        # Create integer indices for MILP (0, 1, 2, ..., m-1)
        self.V = list(range(len(original_vehicle_ids)))
       
        # Create mappings
        self.vehicle_id_to_index = {vid: i for i, vid in enumerate(original_vehicle_ids)}
        self.index_to_vehicle_id = {i: vid for i, vid in enumerate(original_vehicle_ids)}
   
    def _initialize_depot(self):
        # Use negative integers for depot nodes (stations use 0, 1, 2, ...)
        self.source = -1  # Source node (depot/start)
        self.sink = -2  # Sink node (depot/end)
   
    def _initialize_travel_times(self):
      
        # Calculate inter-station travel times
        for i in self.stations:
            for j in self.stations:
                if i == j:
                    # Holding at same station: zero travel time
                    self.T_D[(i, j)] = 0.0
                    self.T_DD[(i, j)] = 1
                else:
                    # Different stations: calculate distance-based travel time
                    station_i = self.state.stations[self.index_to_station_id[i]]
                    station_j = self.state.stations[self.index_to_station_id[j]]
                    
                    distance_km = station_i.distance_to(station_j.get_lat(), station_j.get_lon())
                    travel_time = (distance_km / VEHICLE_SPEED) * 60 + MINUTES_CONSTANT_PER_ACTION
                    
                    self.T_D[(i, j)] = travel_time
                    self.T_DD[(i, j)] = math.ceil(travel_time / self.tau)
        
        # Sink has zero travel time from any station (logical end point)
        for station_idx in self.stations:
            self.T_D[(station_idx, self.sink)] = 0.0
            self.T_DD[(station_idx, self.sink)] = 0.0
            # Source travel times will be set in initialize_source_for_vehicles()
   
    def initialize_source_for_vehicles(self):
        """
        Initialize source node connections in space-time network.
        
        The source node (s) represents the initial state at t=0 and connects to appropriate (i,t) nodes:
        - If vehicle is AT a station: arc from s to (station, t=0) with ZERO travel time
        - If vehicle is EN ROUTE: arc from s to (destination, t=arrival) with remaining travel time
        
        This follows the space-time network formulation where source initializes the subproblem
        and connects it to the last observed state of the system.
        """
        vehicles = list(self.state.vehicles.values())
        #print("HEI vehicles:",vehicles)
        if len(vehicles) > 0:
            vehicle = vehicles[0]  # Assuming single vehicle for now
            current_station_id = vehicle.location.id
            #print(f"HEI Initializing source for vehicle {vehicle.id} at station {current_station_id}")
            
            #print(self.station_id_to_index)
            if current_station_id in self.station_id_to_index:
                current_station_idx = self.station_id_to_index[current_station_id]
                print(f"\n=== INITIALIZING SOURCE NODE (Space-Time Network) ===")
                print(f"Vehicle currently at station {current_station_id} (index {current_station_idx})")
                
                # Only create arc to current station
                self.T_D[(self.source, current_station_idx)] = 0.0
                self.T_DD[(self.source, current_station_idx)] = 1  # Minimum one period
                print(f"  Arc: s -> ({self.index_to_station_id[current_station_idx]}, t=0) [vehicle present]")
                
                # Debug: Print ALL T_DD entries involving source
                print(f"\nAll T_DD entries with source (s={self.source}):")
                source_arcs = [(i, j, periods) for (i, j), periods in self.T_DD.items() if i == self.source]
                for i, j, periods in sorted(source_arcs, key=lambda x: x[1]):
                    if j == self.sink:
                        print(f"  T_DD[({i}, {j})] = {periods}  [s -> sink]")
                    elif j in self.index_to_station_id:
                        print(f"  T_DD[({i}, {j})] = {periods}  [s -> {self.index_to_station_id[j]}]")
                    else:
                        print(f"  T_DD[({i}, {j})] = {periods}  [s -> unknown node {j}]")
                print(f"Total source arcs: {len(source_arcs)}")
                print(f"=== END SOURCE INITIALIZATION ===")
   
    def _initialize_station_inventories(self):
        for station_idx in self.stations:
            station_id = self.index_to_station_id[station_idx]
            station = self.state.stations[station_id]
            # Count available bikes at the station
            self.I_N0[station_idx] = station.number_of_bikes()
   
    def _initialize_target_inventories(self):
        for station_idx in self.stations:
            station_id = self.index_to_station_id[station_idx]
            station = self.state.stations[station_id]
            # Get target state for current day/hour
            self.I_T[station_idx] = station.get_target_state(self.state.day(), self.state.hour())
            
    def _initialize_station_capacities(self):
        for station_idx in self.stations:
            station_id = self.index_to_station_id[station_idx]
            station = self.state.stations[station_id]
            self.Q_S[station_idx] = station.capacity
   
    def _initialize_vehicle_capacities(self):
        for vehicle_idx in self.V:
            vehicle_id = self.index_to_vehicle_id[vehicle_idx]
            vehicle = self.state.vehicles[vehicle_id]
            self.Q_V[vehicle_idx] = vehicle.bike_inventory_capacity
   
    def _initialize_vehicle_loads(self):
        for vehicle_idx in self.V:
            vehicle_id = self.index_to_vehicle_id[vehicle_idx]
            vehicle = self.state.vehicles[vehicle_id]
            self.Q_V0[vehicle_idx] = len(vehicle.get_bike_inventory())
    
    def _initialize_vehicle_destinations(self):
        """
        Initialize eta (η^v) - the initial destination station for each vehicle.
        This is the station where the vehicle is currently located or heading to at t=0.
        Used in constraint (2) to ensure vehicle v departs from source to station eta[v].
        """
        for vehicle_idx in self.V:
            vehicle_id = self.index_to_vehicle_id[vehicle_idx]
            vehicle = self.state.vehicles[vehicle_id]
            vehicle_location_id = vehicle.location.id
            
            if vehicle_location_id in self.station_id_to_index:
                self.eta[vehicle_idx] = self.station_id_to_index[vehicle_location_id]
            else:
                # If vehicle is at depot or unknown location, default to first station
                # This shouldn't happen in normal operation
                self.eta[vehicle_idx] = min(self.stations)
   
    def _initialize_demand(self):
        """
        Initialize demand predictions for each station and time period.
        D[(station_idx, period)] = net demand (arrivals - departures)
       
        Uses arrival/departure intensities from the demand model.
        Positive value = net arrivals (more bikes coming in)
        Negative value = net departures (more bikes leaving)
        
        Note: Includes period 0 (initial time period) through T
        """
        print(self.to_dict().keys())
        for station_idx in self.stations:
            station_id = self.index_to_station_id[station_idx]
            station = self.state.stations[station_id]
            day = self.state.day()
            hour = self.state.hour()

            for period in range(0, self.T + 1):
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
        Initialize maintenance parameters based on station maintenance needs.
        Only allocate time to stations with bikes that need maintenance.
        """
        stations_with_maintenance = []
        
        for station_idx in self.stations:
            station_id = self.index_to_station_id[station_idx]
            station = self.state.stations[station_id]
            
            # Get maintenance statistics
            avg_maint = station.get_average_maintenance_criticality()
            stats = station.get_maintenance_criticality_stats()
            
            # Only allocate maintenance time if station has bikes with some criticality
            if stats['count'] == 0 or avg_maint < 0.05:
                # No bikes or negligible maintenance needs
                self.T_M_min[station_idx] = 0
                self.T_M_max[station_idx] = 0
            else:
                # Calculate how many bikes should be serviced
                # Priority 1: High-criticality bikes (>0.5)
                bikes_to_service = stats['high_criticality_count']
                
                # Priority 2: If no critical bikes but avg > 0.3, service proportionally
                if bikes_to_service == 0 and avg_maint >= 0.3:
                    bikes_to_service = max(1, int(stats['count'] * avg_maint))
                
                # Priority 3: If avg between 0.05-0.3, service at least 1 bike per visit
                # This allows early maintenance before bikes become critical
                if bikes_to_service == 0 and avg_maint >= 0.05:
                    bikes_to_service = max(1, int(stats['count'] * avg_maint * 2))
                
                # Cap by MAX_BIKES_PER_VISIT (realistic constraint)
                bikes_to_service = min(bikes_to_service, MAX_BIKES_PER_VISIT)
                
                # Time = bikes to service × time per bike
                max_time = bikes_to_service * TIME_PER_BIKE_MAINTENANCE
                
                # If maintenance is chosen, must service at least 1 bike (no partial servicing)
                # This ensures either: 0 min (no maintenance) OR at least 3 min (fix 1+ bikes)
                self.T_M_min[station_idx] = TIME_PER_BIKE_MAINTENANCE if max_time > 0 else 0
                self.T_M_max[station_idx] = max_time if max_time > 0 else 0
                
                if max_time > 0:
                    stations_with_maintenance.append((station_id, avg_maint, bikes_to_service, max_time))
        
        # Debug output
        if stations_with_maintenance:
            print(f"\n--- Maintenance Time Allocation ---")
            print(f"Stations with maintenance capacity:")
            for sid, avg, bikes, time in sorted(stations_with_maintenance, key=lambda x: x[1], reverse=True):
                station_idx = self.station_id_to_index[sid]
                t_min = self.T_M_min[station_idx]
                print(f"  {sid}: avg={avg:.3f}, bikes={bikes}, T_M=[{t_min:.1f}, {time:.1f}] min")
        else:
            print(f"\n--- Maintenance Time Allocation ---")
            print(f"No stations allocated maintenance time (all avg_maint < 0.05)")
    
    def initialize_vehicle_ETAs(self):
        """
        Initialize travel times for vehicles that are currently in transit.
        If a vehicle has ETA > 0, it's currently traveling to a location.
        Set the travel time from depot (source) to that location as the remaining time.
        """
        
        for vehicle_idx in self.V:
            vehicle_id = self.index_to_vehicle_id[vehicle_idx]
            vehicle = self.state.vehicles[vehicle_id]
            #print(vehicle.eta)
            if vehicle.eta > 0:
                #print("HEI")
                # Vehicle is in transit - get its destination station
                destination_station_id = vehicle.location.id
                
                # Convert to MILP index
                if destination_station_id in self.station_id_to_index:
                    destination_idx = self.station_id_to_index[destination_station_id]
                    
                    # Calculate remaining travel time in minutes
                    remaining_time = vehicle.eta - self.state.time
                    
                    # Check if remaining time exceeds horizon
                    time_horizon_minutes = self.T * self.tau
                    
                    if remaining_time > time_horizon_minutes:
                        print(f"Vehicle {vehicle_id} remaining time {remaining_time:.2f} > horizon {time_horizon_minutes}. Routing directly to sink.")
                        # Route directly to sink
                        self.eta[vehicle_idx] = self.sink
                        # Set travel time to match horizon
                        self.T_D[(self.source, self.sink)] = float(time_horizon_minutes)
                        self.T_DD[(self.source, self.sink)] = self.T
                    else:
                        # Set travel time from depot (source) to destination
                        self.T_D[(self.source, destination_idx)] = remaining_time
                        
                        # Set discretized travel time (in periods)
                        self.T_DD[(self.source, destination_idx)] = max(1, (remaining_time // self.tau) + 1)
 
 
   
    def to_dict(self):
        """
        Convert parameters to dictionary format for Gurobi model input
        """
        return {
            "T": self.T,
            "tau": self.tau,
            "N": self.stations,
            "s": self.source,
            "d": self.sink,
            "V": self.V,
            "T_D": self.T_D,
            "T_DD": self.T_DD,
            "T_L": self.T_L,
            "T_M_min": self.T_M_min,
            "T_M_max": self.T_M_max,
            "Q_V": self.Q_V,
            "Q_V0": self.Q_V0,
            "eta": self.eta,
            "Q_S": self.Q_S,
            "I_N0": self.I_N0,
            "I_T": self.I_T,
            "D": self.D,
            "w_S": self.w_S,
            "w_C": self.w_C,
            "w_D": self.w_D,
            "r_M": self.r_M
        }
   
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
        
        # Initialize source node based on vehicle locations (must be after _initialize_travel_times)
        self.initialize_source_for_vehicles()
        
        # Initialize vehicle ETAs (for vehicles in transit - must be after travel times)
        self.initialize_vehicle_ETAs()
       
        # Initialize station parameters
        self._initialize_station_inventories()
        self._initialize_station_capacities()
        self._initialize_target_inventories()
       
        # Initialize vehicle parameters
        self._initialize_vehicle_capacities()
        self._initialize_vehicle_loads()
        self._initialize_vehicle_destinations()
       
        # Initialize demand
        self._initialize_demand()
       
        # Initialize maintenance parameters (set to 0 if not using maintenance)
        self._initialize_maintenance()