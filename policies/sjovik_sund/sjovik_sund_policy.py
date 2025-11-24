from policies import Policy
import sim
import math
from policies.sjovik_sund.Sub_problem.subproblem_parameters import (
    MILP_parameters, 
    TIME_PER_BIKE_MAINTENANCE
)
from policies.sjovik_sund.Sub_problem.sjovik_sund_subproblem import run_subproblem_model
from policies.sjovik_sund.visualize_subproblem import Visualizer

"""
IMPORTANT NOTE ON TRAVEL TIMES:
The simulator uses stochastic travel times (lognormal distribution) when traveltime_vehicle_matrix_stddev
is available. This means:
- Policy planning uses EXPECTED travel times (mean of distribution)
- Actual simulation samples from the distribution, causing variation
- Displayed "Expected ETA" is the policy's prediction
- Actual vehicle ETA will differ due to random sampling
This is realistic behavior - real-world travel times vary due to traffic, weather, etc.
"""
 
 
class SjovikSundPolicy(Policy):
    def __init__(self, roaming = False, time_horizon=12, tau=5, weights=None):
        self.roaming = roaming
        self.time_horizon = time_horizon
        self.tau = tau
        self.weights = weights
        self.vehicle_routes = {}  # Track actual routes: {vehicle_id: [(time, station_id), ...]}
        super().__init__()
 
    def get_best_action(self, simul, vehicle):
        # Print state with current time's target states
        day = simul.day()
        hour = simul.hour()
        print(f"\n<State: {len(simul.get_parked_bikes())} bikes in {len(simul.stations)} stations with {len(simul.vehicles)} vehicles>")
        print(f"Current Time: Day {day}, Hour {hour}\n")
        
        # Print stations with maintenance information
        print(f"{'Station':<10} {'Arrive':<8} {'Leave':<8} {'Ideal':<8} {'Bikes':<7} {'AvgMaint':<10}")
        print("-" * 65)
        for station in simul.get_stations():
            target = station.get_target_state(day, hour)
            avg_maint = station.get_average_maintenance_criticality()
            print(f"{station.id:<10} {station.get_arrive_intensity(day, hour):>7.2f} "
                  f"{station.get_leave_intensity(day, hour):>7.2f} {target:>7.1f} "
                  f"{len(station.bikes):>6} {avg_maint:>9.3f}")
        
        # Print maintenance summary
        high_maint_stations = [(s, s.get_average_maintenance_criticality()) 
                               for s in simul.get_stations() 
                               if s.get_average_maintenance_criticality() > 0.3 and len(s.bikes) > 0]
        
        if high_maint_stations:
            high_maint_stations.sort(key=lambda x: x[1], reverse=True)
            print(f"\n Stations with elevated maintenance needs (>0.3):")
            for station, maint in high_maint_stations[:5]:  # Top 5
                stats = station.get_maintenance_criticality_stats()
                print(f"  {station.id}: Avg={maint:.3f}, Max={stats['max']:.3f}, "
                      f"High bikes (>0.5): {stats['high_criticality_count']}/{stats['count']}")
        
        print()
        # Solve subproblem for ALL vehicles based on current system state
        data = MILP_parameters(simul, self.time_horizon, self.weights, self.tau)
        data.initalize_parameters()
        
        # --- Print Simulation Clock ---
        time = simul.time
        day = time // (24*60)
        hour = (time % (24*60)) // 60           
        minute = time % 60
        print(f"\n{'='*60}")
        print(f"SIMULATION CLOCK: {time:.1f} minutes | Day {day}, Time {hour}, minute {minute}")
        print(f"{'='*60}")
        
        # Solve subproblem for ALL vehicles based on current system state
        data = MILP_parameters(simul, self.time_horizon, self.weights, self.tau)
        data.initalize_parameters()
    

        # --- Print Vehicles in Transit ---
        print("\n--- Vehicles in Transit ---")
        vehicles_in_transit = False
        for v_id, v_obj in simul.vehicles.items():
            if v_obj.eta > simul.time:
                vehicles_in_transit = True
                remaining_time = v_obj.eta - simul.time
                
                # Get destination from MILP parameters (expected travel time)
                dest_station_id = v_obj.location.id
                dest_idx = data.station_id_to_index.get(dest_station_id, -1)
                
                expected_time = "N/A"
                if dest_idx != -1:
                    expected_time = data.T_D.get((data.source, dest_idx), "Not in T_D")
                
                print(f"Vehicle {v_id}: -> To {dest_station_id}")
                print(f"  Actual remaining: {remaining_time:.2f} min (stochastic sample)")
                print(f"  Expected: {expected_time} min (policy planning value)")
                if isinstance(expected_time, (int, float)) and abs(remaining_time - expected_time) > 1.0:
                    print(f"  Deviation: {remaining_time - expected_time:+.2f} min (due to stochastic travel times)")
        
        if not vehicles_in_transit:
            print("No vehicles currently in transit.")


        gurobi_output = run_subproblem_model(data.to_dict())
        
        # Check if model found a feasible solution
        if gurobi_output.Status in [3, 4] or gurobi_output.SolCount == 0:
            return sim.Action([], [], [], vehicle.location.id)
        
        # --- NEW: Print All Planned Actions ---
        print("\n--- Planned Actions (Subproblem Solution) ---")
        # Collect all variables with non-zero values
        solution_vars = []
        visited_stations = set()
        
        for var in gurobi_output.getVars():
            if var.x > 0.01: # Filter out zero values
                solution_vars.append((var.varName, var.x))
                
                # Track visited stations for filtering output later
                # Parse variable name to find station indices
                variable = var.varName.strip("]").split("[")
                name = variable[0]
                indices = variable[1].split(',')
                
                if name == 'x':
                    # x[i,j,v,t] -> i and j are stations
                    from_idx = int(indices[0])
                    to_idx = int(indices[1])
                    if from_idx >= 0: visited_stations.add(from_idx)
                    if to_idx >= 0: visited_stations.add(to_idx)
                elif name in ['qL', 'qU', 'lN', 'starv', 'cong', 'dev']:
                    # These usually start with station index i
                    s_idx = int(indices[0])
                    if s_idx >= 0: visited_stations.add(s_idx)

        # Sort for readability (e.g., by variable name then indices)
        solution_vars.sort(key=lambda x: x[0])
        
        # Print Decision Variables (x, qL, qU, qV, tM)
        # Note: m_iv is a binary flag (not a decision) and is omitted for clarity
        print("Decisions:")
        for name, val in solution_vars:
            var_type = name.split("[")[0]
            if var_type in ['x', 'qL', 'qU', 'qV', 'tM', "m_iv"]:
                print(f"  {name} = {val:.2f}")

        
        """
         # Print State Variables (lN, starv, cong, dev) ONLY for visited stations
        print("\nState Variables (Visited Stations Only):")
        # Helper to group variables by type
        state_vars = {'lN': [], 'starv': [], 'cong': [], 'dev': []}
        # Print state variables only for visited stations
        for name, val in solution_vars:
            var_type = name.split("[")[0]
            if var_type in state_vars:
                # Check if this variable belongs to a visited station
                indices = name.strip("]").split("[")[1].split(',')
                s_idx = int(indices[0])
                if s_idx in visited_stations:
                    state_vars[var_type].append(f"{name}={val:.2f}")

        # Print horizontally
        for var_type, values in state_vars.items():
            if values:
                print(f"  {var_type}: {', '.join(values)}")       
        """


        print("---------------------------------------------")

        # Extract this vehicle's action from the multi-vehicle solution
        next_station, bikes_to_pickup, bikes_to_deliver, maintenance_time = self.return_solution(gurobi_output, vehicle, data)
        
        # Track actual route - record current position before moving
        if vehicle.id not in self.vehicle_routes:
            self.vehicle_routes[vehicle.id] = []
        
        # Record current position and time
        self.vehicle_routes[vehicle.id].append((simul.time, vehicle.location.id))
        
        # SISTE ENDRINGER HER 
        # Record metrics for deliveries and pickups
        simul.metrics.add_aggregate_metric(simul, 'num bike pickups', len(bikes_to_pickup))
        simul.metrics.add_aggregate_metric(simul, 'num bike deliveries', len(bikes_to_deliver))
        simul.metrics.add_aggregate_metric(simul, 'vehicle arrivals', 1)
        
        # --- DEBUG: Print current station status ---
        print(f"\n--- STATION STATUS CHECK ---")
        for station_id, station_obj in simul.stations.items():
            bikes_count = station_obj.number_of_bikes()
            capacity = station_obj.capacity
            utilization = bikes_count / capacity if capacity > 0 else 0
            status = "FULL" if bikes_count >= capacity else "EMPTY" if bikes_count == 0 else "OK"
            print(f"  {station_id}: {bikes_count}/{capacity} bikes ({utilization:.1%}) [{status}]")
        



        # --- NEW: Adjusted Print Sentences ---
        # 1. Current Actions at Station
        maint_str = f", Maintenance: {maintenance_time:.2f} min" if maintenance_time > 0 else ", No maintenance"
        print(f"\nVehicle {vehicle.id} at {vehicle.location.id}: Picking up {len(bikes_to_pickup)} bikes, Delivering {len(bikes_to_deliver)} bikes{maint_str}.")
        
        # 2. Next Movement
        expected_travel_time = data.T_D.get((data.station_id_to_index[vehicle.location.id], data.station_id_to_index[next_station]), 0.0)
        print(f"Vehicle {vehicle.id} is going to station {next_station} next.")
        print(f"  Expected travel time: {expected_travel_time:.2f} minutes (policy planning value)")
        print(f"  Expected ETA: minute {simul.time + expected_travel_time:.2f}")
        print(f"  Note: Actual travel time may vary due to stochastic effects in simulation")
            
        return sim.Action(
            [],               # batteries to swap
            bikes_to_pickup, #list of bike id's
            bikes_to_deliver, #list of bike id's
            next_station, #id
            maintenance_time=maintenance_time  # Include maintenance time from MILP solution
        )
 
      
    def return_solution(self, gurobi_output, vehicle, data):
        # Initialize variables
        first_move_period = 1000
        loading_quantity = unloading_quantity = maintenance_time = 0.0
        loading_ids, unloading_ids = [], []
        station_id = vehicle.location.id
        vehicle_idx = data.vehicle_id_to_index[vehicle.id]
        current_station_idx = data.station_id_to_index[vehicle.location.id]
        print(f"\n=== ROUTING ANALYSIS for Vehicle {vehicle.id} (Index {vehicle_idx}) ===")
        #SISTE ENDRINGER HER
        # 1. Extract and print route sequence
        print(f"Route Sequence:")
        route_sequence = []
        
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name, indices = variable[0], variable[1].split(',')
            
            if name == 'x' and round(var.x, 0) == 1 and int(indices[2]) == vehicle_idx:
                from_idx, to_idx, period = int(indices[0]), int(indices[1]), int(indices[3])
                
                # Format station names
                from_name = data.index_to_station_id.get(from_idx, "Source" if from_idx == -1 else "Sink" if from_idx == -2 else str(from_idx))
                to_name = data.index_to_station_id.get(to_idx, "Source" if to_idx == -1 else "Sink" if to_idx == -2 else str(to_idx))
                route_sequence.append((period, from_name, to_name, from_idx, to_idx))

                # Find first move from current station to a different station
                if from_idx == current_station_idx and to_idx >= 0 and to_idx != current_station_idx and period < first_move_period:
                    first_move_period = period
                    station_id = data.index_to_station_id[to_idx]

        # Print route sequence
        for period, from_name, to_name, _, _ in sorted(route_sequence, key=lambda x: x[0]):
            print(f"  Period {period}: {from_name} -> {to_name}")
        if not route_sequence:
            print("  (No movement from current station found in solution)")

        # 2. Extract actions at current station
        print(f"\nActions at Current Station ({vehicle.location.id}):")
        
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name, indices = variable[0], variable[1].split(',')
            
            if name in ['qL', 'qU', 'tM'] and var.x > 0.01:
                s_idx, v_idx, period_t = int(indices[0]), int(indices[1]), int(indices[2])
                
                if v_idx == vehicle_idx and s_idx == current_station_idx and period_t <= first_move_period:
                    if name == 'qL':
                        loading_quantity += var.x
                        print(f"  Load: {var.x:.2f} (Period {period_t})")
                    elif name == 'qU':
                        unloading_quantity += var.x
                        print(f"  Unload: {var.x:.2f} (Period {period_t})")
                    elif name == 'tM':
                        maintenance_time += var.x
                        print(f"  Maintenance: {var.x:.2f} minutes (Period {period_t})")
        
        if loading_quantity == 0 and unloading_quantity == 0 and maintenance_time == 0:
            print("  (No loading/unloading/maintenance actions)")

        # 3. Select bikes to pickup/deliver
        bikes_at_station = list(vehicle.location.bikes.values())
        bikes_at_vehicle = vehicle.get_bike_inventory()
        
        # Round up pickups, round down deliveries
        num_to_pickup = min(len(bikes_at_station), math.ceil(loading_quantity))
        num_to_deliver = min(len(bikes_at_vehicle), math.floor(unloading_quantity))
        
        unloading_ids = [bikes_at_vehicle[i].bike_id for i in range(num_to_deliver)]
        loading_ids = [bikes_at_station[i].bike_id for i in range(num_to_pickup)]
        
        print(f"\nBike Transfer Summary:")
        print(f"  Pickup: {num_to_pickup} bikes (Target: {loading_quantity:.2f})")
        print(f"  {loading_ids}")
        print(f"  Deliver: {num_to_deliver} bikes (Target: {unloading_quantity:.2f})")
        print(f"  {unloading_ids}")
        print(f"  Maintenance: {maintenance_time:.2f} minutes")
        
        return station_id, loading_ids, unloading_ids, maintenance_time
    
    def write_routes_to_file(self, seed=None):
        """Write the actual routes taken by all vehicles to a file"""
        import os
        
        # Create results directory if it doesn't exist
        results_dir = './policies/sjovik_sund/simulation_results/'
        os.makedirs(results_dir, exist_ok=True)
        
        # Create filename with seed if provided
        if seed is not None:
            filename = f'simulation_result_route_seed_{seed}.txt'
        else:
            filename = 'simulation_result_route.txt'
        
        filepath = results_dir + filename
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ACTUAL VEHICLE ROUTES (SIMULATION COMPLETE)\n")
            f.write("="*80 + "\n\n")
            
            for vehicle_id, route in self.vehicle_routes.items():
                f.write(f"Vehicle {vehicle_id} Route:\n")
                if not route:
                    f.write("  No route recorded\n\n")
                    continue
                
                # Sort by time to ensure chronological order
                route.sort(key=lambda x: x[0])
                
                # Write route with time information
                for i, (time, station_id) in enumerate(route):
                    # Convert time to day/hour/minute format
                    day = int(time // (24*60))
                    hour = int((time % (24*60)) // 60)
                    minute = int(time % 60)
                    
                    if i == 0:
                        f.write(f"  Start: {station_id} at Day {day}, Hour {hour:02d}:{minute:02d} (t={time:.1f})\n")
                    else:
                        # Calculate travel time from previous station
                        prev_time = route[i-1][0]
                        travel_duration = time - prev_time
                        f.write(f"  Move {i}: {station_id} at Day {day}, Hour {hour:02d}:{minute:02d} (t={time:.1f}) [+{travel_duration:.1f} min]\n")
                
                # Summary statistics
                total_time = route[-1][0] - route[0][0] if len(route) > 1 else 0
                unique_stations = len(set(station for _, station in route))
                f.write(f"  Summary: {len(route)} stops, {unique_stations} unique stations, {total_time:.1f} min total\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"Vehicle routes written to: {filepath}")  
 
