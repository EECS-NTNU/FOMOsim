from policies import Policy
import sim
import math
from policies.sjovik_sund.Sub_problem.subproblem_parameters import MILP_parameters
from policies.sjovik_sund.Sub_problem.sjovik_sund_subproblem import run_subproblem_model
 
 
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
        gurobi_output = run_subproblem_model(data.to_dict())
        
        # Check if model found a feasible solution
        if gurobi_output.Status in [3, 4] or gurobi_output.SolCount == 0:
            print(f"⚠ WARNING: Model infeasible or no solution - vehicle {vehicle.id} stays at current location")
            return sim.Action([], [], [], vehicle.location.id)
        
        # Extract this vehicle's action from the multi-vehicle solution
        next_station, bikes_to_pickup, bikes_to_deliver = self.return_solution(gurobi_output, vehicle, data)
        print(f"Vehicle {vehicle.id} going to station {next_station} to pick up {len(bikes_to_pickup)} bikes and deliver {len(bikes_to_deliver)} bikes.")
            
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
       
        print(f"\n=== DEBUGGING ARC SELECTION for vehicle {vehicle.id} (index {vehicle_idx}) at station {vehicle.location.id} (index {current_station_idx}) ===")
        
        # Print ALL x variables that are set to 1 for this vehicle
        print(f"\nAll arcs (x variables = 1) for vehicle {vehicle_idx}:")
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
                    print(f"  x[{from_idx},{to_idx},{v_idx},{period}] = 1")
        
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name = variable[0]
            indices = variable[1].split(',')
            # Model uses x[i,j,v,t] where i=from, j=to, v=vehicle, t=period
            if name == 'x' and int(indices[2]) == vehicle_idx and round(var.x,0) == 1:
                from_idx = int(indices[0])
                to_idx = int(indices[1])
                period = int(indices[3])
                
                # Only consider routes starting from current location
                if from_idx == current_station_idx and to_idx >= 0 and to_idx != current_station_idx:
                    if period < first_move_period:
                        first_move_period = period
                        destination_station_idx = to_idx  # Save the destination index!
                        # Map back to actual station ID
                        station_id = data.index_to_station_id[to_idx]
                        print(f"Found routing: from index {from_idx} -> to index {to_idx} (station {station_id}) at period {period}")
 
        # Debug: Print ALL qL and qU values for this vehicle at CURRENT station
        print(f"\n=== DEBUGGING qL/qU for vehicle {vehicle_idx} at CURRENT station {current_station_idx} ===")
        print(f"First move period: {first_move_period}")
        
        # Show all qL/qU at current station
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name = variable[0]
            if (name == 'qL' or name == 'qU') and var.x > 0.01:
                indices = variable[1].split(',')
                v_idx = int(indices[1])
                s_idx = int(indices[0])
                t_idx = int(indices[2])
                # Look at CURRENT station where vehicle will load/unload before moving
                if v_idx == vehicle_idx and s_idx == current_station_idx:
                    print(f"  {name}[{s_idx},{v_idx},{t_idx}] = {var.x:.2f}")
        
        # Extract loading/unloading at CURRENT station (where vehicle is now)
        # Vehicle will load/unload HERE before moving to next station
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name = variable[0]
            # Only process qL and qU variables
            if name not in ['qL', 'qU']:
                continue
            indices = variable[1].split(',')
            # Model uses qL[i,v,t] and qU[i,v,t] where i=station, v=vehicle, t=period
            # Look at current_station_idx (where vehicle currently is) AND only in the departure period
            period_t = int(indices[2])
            if name == 'qL' and int(indices[1]) == vehicle_idx and var.x > 0.01 and int(indices[0]) == current_station_idx and period_t == first_move_period:
                loading_quantity += var.x
                print(f"✓ Loading {var.x} bikes at CURRENT station index {indices[0]} in period {indices[2]} (departure period)")
            elif name == 'qU' and int(indices[1]) == vehicle_idx and var.x > 0.01 and int(indices[0]) == current_station_idx and period_t == first_move_period:
                unloading_quantity += var.x
                print(f"✓ Unloading {var.x} bikes at CURRENT station index {indices[0]} in period {indices[2]} (departure period)")
        
        print(f"\nTotal loading_quantity: {loading_quantity}, unloading_quantity: {unloading_quantity}")
        
        # OLD CODE (commented out - had rounding issues with fractional bikes)
        # if not (loading_quantity == 0 and unloading_quantity == 0):
        #     bikes_at_station = list(vehicle.location.bikes.values())
        #     bikes_at_vehicle = vehicle.get_bike_inventory()
        #     print(f"Will pick up: min({len(bikes_at_station)}, {int(loading_quantity)}) = {min(len(bikes_at_station), int(loading_quantity))}")
        #     print(f"Will deliver: min({len(bikes_at_vehicle)}, {int(unloading_quantity)}) = {min(len(bikes_at_vehicle), int(unloading_quantity))}")
        #     for bike in range(0, min(len(bikes_at_station), int(loading_quantity))):
        #         loading_ids.append(bikes_at_station[bike].bike_id)
        #     for bike in range(0,min(len(bikes_at_vehicle), int(unloading_quantity))):
        #         unloading_ids.append(bikes_at_vehicle[bike].bike_id)
        
        # NEW CODE: Use ceiling for pickups (round up), floor for deliveries (round down)
        # This ensures we respect the model's fractional quantities properly
        bikes_at_station = list(vehicle.location.bikes.values()) #creates list of bike objects
        bikes_at_vehicle = vehicle.get_bike_inventory()
        
        # Show bike IDs BEFORE loading/unloading
        bike_ids_on_vehicle_before = [bike.bike_id for bike in bikes_at_vehicle]
        print(f"\n=== BEFORE Loading/Unloading ===")
        print(f"Vehicle has {len(bike_ids_on_vehicle_before)} bikes: {bike_ids_on_vehicle_before}")
        print(f"Station has {len(bikes_at_station)} bikes available")
        print(f"Vehicle capacity: {vehicle.bike_inventory_capacity}")
        
        # Round up pickups (be aggressive about loading), round down deliveries (conservative about unloading)
        num_to_pickup = min(len(bikes_at_station), math.ceil(loading_quantity))
        num_to_deliver = min(len(bikes_at_vehicle), math.floor(unloading_quantity))
        
        print(f"\nWill pick up: min({len(bikes_at_station)}, ceil({loading_quantity:.2f})) = {num_to_pickup}")
        print(f"Will deliver: min({len(bikes_at_vehicle)}, floor({unloading_quantity:.2f})) = {num_to_deliver}")
        
        # Perform unloading first
        for bike in range(0, num_to_deliver):
            unloading_ids.append(bikes_at_vehicle[bike].bike_id)
            print(f"  Unloading bike ID {bikes_at_vehicle[bike].bike_id}")
        
        # Then perform loading
        for bike in range(0, num_to_pickup):
            loading_ids.append(bikes_at_station[bike].bike_id)
            print(f"  Loading bike ID {bikes_at_station[bike].bike_id}")
        
        # Show bike IDs AFTER loading/unloading (predicted)
        print(f"\n=== AFTER Loading/Unloading (predicted) ===")
        predicted_bikes_on_vehicle = [b for b in bike_ids_on_vehicle_before if b not in unloading_ids] + loading_ids
        print(f"Vehicle will have {len(predicted_bikes_on_vehicle)} bikes: {predicted_bikes_on_vehicle}")
 
        print(f"\nDecision: Go to {station_id}, pickup {len(loading_ids)} bikes, deliver {len(unloading_ids)} bikes")
        
        return station_id, loading_ids, unloading_ids  
 