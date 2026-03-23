from policies import Policy
import sim
import math
from policies.sjovik_sund.sub_problem.subproblem_parameters import (
    MILP_parameters
)
from policies.sjovik_sund.sub_problem.sjovik_sund_subproblem import run_subproblem_model
#from policies.sjovik_sund.scripts.route_visualization.visualize_subproblem import Visualizer


class SjovikSundPolicy(Policy):
    def __init__(self, roaming = False, time_horizon=12, tau=5, weights=None, hour_from=7, hour_to=23, maintenance_enabled=True):
        self.roaming = roaming
        self.time_horizon = time_horizon
        self.tau = tau
        self.weights = weights
        self.vehicle_routes = {}  # Track actual routes: {vehicle_id: [(time, station_id), ...]}
        self.optimality_gaps = []
        super().__init__(maintenance_enabled=maintenance_enabled)
        self.set_time_of_service(hour_from=hour_from, hour_to=hour_to)  # Set working hours (default 7 AM - 4 PM)
 
    def get_best_action(self, simul, vehicle):
        import time as time_module
        t_action_start = time_module.time()

        # NB! Print below will reflect the state before optimization
        
        # Print stations with maintenance information
        '''
        print(f"{'Station':<10} {'Arrive':<8} {'Leave':<8} {'Ideal':<8} {'Bikes':<7} {'AvgMaint':<10}")
        print("-" * 65)
        for station in simul.get_stations():
            target = station.get_target_state(day, hour)
            avg_maint = station.get_average_maintenance_criticality()
            print(f"{station.id:<10} {station.get_arrive_intensity(day, hour):>7.2f} "
                  f"{station.get_leave_intensity(day, hour):>7.2f} {target:>7.1f} "
                  f"{len(station.bikes):>6} {avg_maint:>9.3f}")        
        '''
        
        # Print maintenance summary
        '''high_maint_stations = [(s, s.get_average_maintenance_criticality()) 
                               for s in simul.get_stations() 
                               if s.get_average_maintenance_criticality() > 0.3 and len(s.bikes) > 0]
        
        if high_maint_stations:
            high_maint_stations.sort(key=lambda x: x[1], reverse=True)
            print(f"\n Stations with elevated maintenance needs (>0.3):")
            for station, maint in high_maint_stations[:5]:  # Top 5
                stats = station.get_maintenance_criticality_stats()
                print(f"  {station.id}: Avg={maint:.3f}, Max={stats['max']:.3f}, "
                      f"High bikes (>0.5): {stats['high_criticality_count']}/{stats['count']}")'''
        
        # [DIAGNOSTICS] Calculate inventory imbalance diagnostics
        '''
        total_imbalance = 0
        max_imbalance = 0
        for station in simul.get_stations():
            target = station.get_target_state(day, hour)
            current = len(station.bikes)
            imbalance = abs(current - target)
            total_imbalance += imbalance
            max_imbalance = max(max_imbalance, imbalance)
        avg_imbalance = total_imbalance / len(simul.get_stations())
        print(f"[DIAGNOSTICS] Inventory imbalance: avg={avg_imbalance:.2f}, max={max_imbalance}, total={total_imbalance}")
        '''

        # --- Print Simulation Clock ---
        # Print state with current time's target states

        time = simul.time        
        day = simul.day()
        hour = simul.hour()
        minute = time % 60
        print(f"\n{'='*60}")
        print(f"STARTING OPTIMIZATION - SIMULATION CLOCK: {time:.1f} minutes | Day {day}, Hour {hour}, Minute {minute}")
        print(f"{'='*60}")
        
        # Solve subproblem for ALL vehicles based on current system state
        data = MILP_parameters(simul, self.time_horizon, self.weights, self.tau)
        data.initalize_parameters()


        gurobi_output, gap = run_subproblem_model(data.to_dict())
        
        if gap is not None:
            self.optimality_gaps.append(gap)
        
        # Check if model found a feasible solution
        if gurobi_output.Status in [3, 4] or gurobi_output.SolCount == 0:
            return sim.Action([], [], [], vehicle.location.id)
        
        # Visualize the solution - KOMMENTER UT FOR Å UNNGÅ VISUALISERING
        '''try:
            vis = Visualizer(gurobi_output, data)
            vis.visualize_route()
        except Exception as e:
            print(f"Visualization failed: {e}")'''
        
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
                elif name in ['qL', 'qU', 'qL_curr', 'qL_other', 'qU_curr', 'qU_other', 'lN', 'starv', 'cong', 'dev']:
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
            if var_type in ['x', 'qL', 'qU', 'qL_curr', 'qL_other', 'qU_curr', 'qU_other', 'qV', 'tM', "m_iv"]:
                print(f"  {name} = {val:.2f}")


        # Extract this vehicle's action from the multi-vehicle solution
        next_station, bikes_to_pickup, bikes_to_deliver, maintenance_time, route_sequence = self.return_solution(gurobi_output, vehicle, data)
        
        # Track actual route - record current position before moving
        if vehicle.id not in self.vehicle_routes:
            self.vehicle_routes[vehicle.id] = []
        
        # Record current position and time
        self.vehicle_routes[vehicle.id].append((simul.time, vehicle.location.id))
        
        # NOTE: Metrics for pickups/deliveries are now logged in State.do_action()
        # for consistency across all policies
        simul.metrics.add_aggregate_metric(simul, 'vehicle arrivals', 1)
        simul.metrics.add_aggregate_metric(simul, 'maintenance time', maintenance_time)
        
        # Print comprehensive action summary
        print(f"\n=== ACTION SUMMARY for Vehicle {vehicle.id} ===")
        print(f"Current Station: {vehicle.location.id}")
        
        # Print route sequence
        print(f"\nRoute Sequence:")
        if route_sequence:
            for period, from_name, to_name, _, _ in sorted(route_sequence, key=lambda x: x[0]):
                print(f"  Period {period}: {from_name} -> {to_name}")
        else:
            print("  (No movement from current station found in solution)")
        
        print(f"\nActions: Pickup {len(bikes_to_pickup)} bikes, Deliver {len(bikes_to_deliver)} bikes" + 
              (f", Maintenance: {maintenance_time:.2f} min" if maintenance_time > 0 else ""))
        if bikes_to_pickup:
            print(f"  Pickup IDs: {bikes_to_pickup}")
        if bikes_to_deliver:
            print(f"  Deliver IDs: {bikes_to_deliver}")
        
        expected_travel_time = data.T_D.get((data.station_id_to_index[vehicle.location.id], data.station_id_to_index[next_station]), 0.0)
        print(f"Next Station: {next_station} (Expected travel time: {expected_travel_time:.2f} min, ETA: {simul.time + expected_travel_time:.2f})")
        print(f"{'='*50}\n")
            
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
        next_station_id = vehicle.location.id
        vehicle_idx = data.vehicle_id_to_index[vehicle.id]
        current_station_idx = data.station_id_to_index[vehicle.location.id]

        # Extract route sequence
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
                    next_station_id = data.index_to_station_id[to_idx]

        # Extract actions at current station
        
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name, indices = variable[0], variable[1].split(',')
            
            # Handle all variable naming schemes: qL/qU (old), qL_curr/qL_other/qU_curr/qU_other (new)
            if name in ['qL', 'qU', 'qL_curr', 'qL_other', 'qU_curr', 'qU_other', 'tM'] and var.x > 0.01:
                # All current naming schemes use [i,v,t] with 3 indices
                s_idx, v_idx, period_t = int(indices[0]), int(indices[1]), int(indices[2])
                
                if v_idx == vehicle_idx and s_idx == current_station_idx and period_t <= first_move_period:
                    if name in ['qL', 'qL_curr', 'qL_other']:
                        loading_quantity += var.x
                    elif name in ['qU', 'qU_curr', 'qU_other']:
                        unloading_quantity += var.x
                    elif name == 'tM':
                        maintenance_time += var.x 
        # 3. Select bikes to pickup/deliver
        net_transfer = loading_quantity - unloading_quantity

        bikes_at_station = list(vehicle.location.bikes.values())
        bikes_at_vehicle = vehicle.get_bike_inventory()
        
        # Calculate vehicle's remaining capacity
        vehicle_remaining_capacity = vehicle.bike_inventory_capacity - len(bikes_at_vehicle)

        if net_transfer > 0: #net pickup
            # Must respect both station availability AND vehicle capacity
            num_to_pickup = min(len(bikes_at_station), math.ceil(net_transfer), vehicle_remaining_capacity)
            num_to_deliver = 0
            loading_ids = [bikes_at_station[i].bike_id for i in range(num_to_pickup)]
            unloading_ids = []
        elif net_transfer < 0: #net delivery
            num_to_deliver = min(len(bikes_at_vehicle), math.floor(-net_transfer))
            num_to_pickup = 0
            unloading_ids = [bikes_at_vehicle[i].bike_id for i in range(num_to_deliver)]
            loading_ids = []
        else: #no change
            num_to_pickup = num_to_deliver = 0
            loading_ids = []
            unloading_ids = []
        
        return next_station_id, loading_ids, unloading_ids, maintenance_time, route_sequence
    