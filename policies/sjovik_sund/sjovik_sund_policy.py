from policies import Policy
import sim
import math
from policies.sjovik_sund.Sub_problem.subproblem_parameters import MILP_parameters
from policies.sjovik_sund.Sub_problem.sjovik_sund_subproblem import run_subproblem_model
from policies.sjovik_sund.visualize_subproblem import Visualizer
 
 
class SjovikSundPolicy(Policy):
    def __init__(self, roaming = False, time_horizon=12, tau=5, weights=None):
        self.roaming = roaming
        self.time_horizon = time_horizon
        self.tau = tau
        self.weights = weights
        super().__init__()
 
    def get_best_action(self, simul, vehicle):
        # Solve subproblem for ALL vehicles based on current system state
        data = MILP_parameters(simul, self.time_horizon, self.weights, self.tau)
        data.initalize_parameters()
        
        # --- NEW: Print Simulation Clock ---
        time = simul.time
        day = time // (24*60)
        hour = (time % (24*60)) // 60           
        minute = time % 60
        print(f"\n{'='*60}")
        print(f"SIMULATION CLOCK: Day {day}, Hour {hour}, Minute {minute}")
        print(f"{'='*60}")

        # --- NEW: Print Vehicles in Transit ---
        print("\n--- Vehicles in Transit ---")
        vehicles_in_transit = False
        for v_id, v_obj in simul.vehicles.items():
            if v_obj.eta > simul.time:
                vehicles_in_transit = True
                #driven_time = simul.time - v_obj.departure_time if hasattr(v_obj, 'departure_time') else "N/A"
                remaining_time = v_obj.eta - simul.time
                
                # Get destination from MILP parameters (T_D from source)
                # Note: We need to find the destination index first
                dest_station_id = v_obj.location.id
                dest_idx = data.station_id_to_index.get(dest_station_id, -1)
                
                milp_remaining = "N/A"
                if dest_idx != -1:
                    milp_remaining = data.T_D.get((data.source, dest_idx), "Not in T_D")
                
                print(f"Vehicle {v_id}: Remaining {remaining_time:.2f} min (MILP T_D: {milp_remaining}) -> To {dest_station_id}")
        
        if not vehicles_in_transit:
            print("No vehicles currently in transit.")

        gurobi_output = run_subproblem_model(data.to_dict())
        
        # Check if model found a feasible solution
        if gurobi_output.Status in [3, 4] or gurobi_output.SolCount == 0:
            print(f"⚠ WARNING: Model infeasible or no solution - vehicle {vehicle.id} stays at current location")
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
        
        # Print Decision Variables (x, qL, qU, qV, tM, m_iv)
        print("Decisions:")
        for name, val in solution_vars:
            var_type = name.split("[")[0]
            if var_type in ['x', 'qL', 'qU', 'qV', 'tM', 'm_iv']:
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
        next_station, bikes_to_pickup, bikes_to_deliver = self.return_solution(gurobi_output, vehicle, data)
        
        # --- NEW: Adjusted Print Sentences ---
        # 1. Current Action
        print(f"\nVehicle {vehicle.id} at {vehicle.location.id}: Picking up {len(bikes_to_pickup)} bikes, Delivering {len(bikes_to_deliver)} bikes.")
        
        # 2. Next Movement
        travel_time = data.T_D.get((data.station_id_to_index[vehicle.location.id], data.station_id_to_index[next_station]), 0.0)
        print(f"Vehicle {vehicle.id} is going to station {next_station} next. The trip should take {travel_time:.2f} minutes. Estimated arrival at minute {simul.time + travel_time:.2f}.")
            
        return sim.Action(
            [],               # batteries to swap
            bikes_to_pickup, #list of bike id's
            bikes_to_deliver, #list of bike id's
            next_station, #id
        )  
 
      
    def return_solution(self, gurobi_output, vehicle, data):
        first_move_period = 1000
        loading_quantity = 0
        unloading_quantity = 0
        loading_ids = []
        unloading_ids = []
        station_id = vehicle.location.id
       
        # Get MILP indices
        vehicle_idx = data.vehicle_id_to_index[vehicle.id]
        current_station_idx = data.station_id_to_index[vehicle.location.id]

        ## SISTE KOMMENTAR FØR ENDRING 
       
        print(f"\n=== ROUTING ANALYSIS for Vehicle {vehicle.id} (Index {vehicle_idx}) ===")
        
        # 1. Print Route Sequence
        print(f"Route Sequence:")
        route_found = False
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name = variable[0]
            if name == 'x' and round(var.x, 0) == 1:
                indices = variable[1].split(',')
                v_idx = int(indices[2])
                if v_idx == vehicle_idx:
                    from_idx = int(indices[0])
                    to_idx = int(indices[1])
                    period = int(indices[3])
                    
                    # Format station names
                    from_name = data.index_to_station_id.get(from_idx, "Source" if from_idx == -1 else "Sink" if from_idx == -2 else str(from_idx))
                    to_name = data.index_to_station_id.get(to_idx, "Source" if to_idx == -1 else "Sink" if to_idx == -2 else str(to_idx))
                    
                    print(f"  Period {period}: {from_name} -> {to_name}")

                    # Check for next move
                    if from_idx == current_station_idx and to_idx >= 0 and to_idx != current_station_idx:
                        if period < first_move_period:
                            first_move_period = period
                            station_id = data.index_to_station_id[to_idx]
                            route_found = True

        if not route_found:
            print("  (No movement from current station found in solution)")

        # 2. Print Current Station Actions
        print(f"\nActions at Current Station ({vehicle.location.id}):")
        
        # Extract loading/unloading at CURRENT station
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name = variable[0]
            if name not in ['qL', 'qU']: continue
            
            indices = variable[1].split(',')
            s_idx = int(indices[0])
            v_idx = int(indices[1])
            period_t = int(indices[2])
            
            if v_idx == vehicle_idx and s_idx == current_station_idx and period_t == first_move_period:
                if name == 'qL' and var.x > 0.01:
                    loading_quantity += var.x
                    print(f"  Load: {var.x:.2f} (Period {period_t})")
                elif name == 'qU' and var.x > 0.01:
                    unloading_quantity += var.x
                    print(f"  Unload: {var.x:.2f} (Period {period_t})")
        
        if loading_quantity == 0 and unloading_quantity == 0:
            print("  (No loading/unloading actions)")

        # 3. Bike Selection Logic
        bikes_at_station = list(vehicle.location.bikes.values())
        bikes_at_vehicle = vehicle.get_bike_inventory()
        
        # Round up pickups, round down deliveries
        num_to_pickup = min(len(bikes_at_station), math.ceil(loading_quantity))
        num_to_deliver = min(len(bikes_at_vehicle), math.floor(unloading_quantity))
        
        # Perform unloading first
        for bike in range(0, num_to_deliver):
            unloading_ids.append(bikes_at_vehicle[bike].bike_id)
        
        # Then perform loading
        for bike in range(0, num_to_pickup):
            loading_ids.append(bikes_at_station[bike].bike_id)
        
        print(f"\nBike Transfer Summary:")
        print(f"  Pickup: {num_to_pickup} bikes (Target: {loading_quantity:.2f})")
        print(f"  Deliver: {num_to_deliver} bikes (Target: {unloading_quantity:.2f})")
        
        return station_id, loading_ids, unloading_ids  
 