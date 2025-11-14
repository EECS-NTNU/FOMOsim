from policies import Policy
import sim
from policies.sjovik_sund.Sub_problem.subproblem_parameters import MILP_parameters
from policies.sjovik_sund.Sub_problem.sjovik_sund_subproblem import run_subproblem_model
 
 
class SjovikSundPolicy(Policy):
    def __init__(self, roaming = False, time_horizon=25, tau=5, weights=None):
        self.roaming = roaming
        self.time_horizon = time_horizon
        self.tau = tau
        self.weights = weights
        super().__init__()
 
    def get_best_action(self, simul, vehicle):
        print(f"\n{'='*80}")
        print(f"🚗 VEHICLE {vehicle.id} DECISION at time {simul.time} (hour {simul.hour()})")
        print(f"   Location: {vehicle.location.id if vehicle.location else 'None'}")
        print(f"   Current load: {len(vehicle.get_bike_inventory())} bikes")
        print(f"{'='*80}")
        
        # Print current state of all stations
        print(f"\n📊 CURRENT STATION STATES:")
        print(f"{'Station':<10} {'Bikes':<8} {'Capacity':<10} {'Fill %':<10} {'Target':<10} {'Deviation':<10}")
        print(f"{'-'*70}")
        stations = simul.get_stations()
        day = int((simul.time // (24 * 60)) % 7)
        hour = int((simul.time // 60) % 24)
        for st in stations:
            bikes = len(st.bikes)
            capacity = st.capacity
            fill_pct = (bikes / capacity * 100) if capacity > 0 else 0
            target = st.target_state[day][hour] if hasattr(st, 'target_state') else 0
            deviation = bikes - target
            print(f"{st.id:<10} {bikes:<8} {capacity:<10} {fill_pct:<10.1f} {target:<10.1f} {deviation:>+10.1f}")
        print(f"{'-'*70}\n")
       
        data = MILP_parameters(simul, self.time_horizon, self.weights, self.tau)
        data.initalize_parameters()
        gurobi_output = run_subproblem_model(data.to_dict())
       
        # Check if model found a solution
        # Status codes: 2=OPTIMAL, 9=TIME_LIMIT, 11=INTERRUPTED
        # Accept any solution where Gurobi found at least one feasible solution
        if gurobi_output.status not in [2, 9, 11]:  # Not optimal, time limit, or interrupted
            print(f"Warning: Model status = {gurobi_output.status} for vehicle {vehicle.id} - no feasible solution found")
            # Return do-nothing action
            return sim.Action([], [], [], vehicle.location.id)
        
        # Check if any solution was found
        if gurobi_output.SolCount == 0:
            print(f"Warning: No solution found for vehicle {vehicle.id}")
            return sim.Action([], [], [], vehicle.location.id)
        
        if gurobi_output.status != 2:
            print(f"Note: Using best solution found (status={gurobi_output.status}, gap={gurobi_output.MIPGap*100:.2f}%)")
       
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
       
        # Debug: Print vehicle info
        print(f"\n--- Extracting solution for vehicle {vehicle.id} (index {vehicle_idx}) ---")
        print(f"Current location: {vehicle.location.id} (index {current_station_idx})")
        print(f"Current load: {len(vehicle.get_bike_inventory())} bikes")
       
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
                        # Map back to actual station ID
                        station_id = data.index_to_station_id[to_idx]
                        print(f"Found routing: from index {from_idx} -> to index {to_idx} (station {station_id}) at period {period}")
 
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name = variable[0]
            indices = variable[1].split(',')
            # Model uses qL[i,v,t] and qU[i,v,t] where i=station, v=vehicle, t=period
            if name == 'qL' and int(indices[1]) == vehicle_idx and round(var.x,0) > 0 and int(indices[0]) == current_station_idx and int(indices[2]) <= first_move_period:
                loading_quantity += var.x
                print(f"Loading {var.x} bikes at station index {indices[0]} in period {indices[2]}")
            elif name == 'qU' and int(indices[1]) == vehicle_idx and round(var.x,0) > 0 and int(indices[0]) == current_station_idx and int(indices[2]) <= first_move_period:
                unloading_quantity += var.x
                print(f"Unloading {var.x} bikes at station index {indices[0]} in period {indices[2]}")
               
        if not (loading_quantity == 0 and unloading_quantity == 0):
            bikes_at_station = list(vehicle.location.bikes.values()) #creates list of bike objects
            bikes_at_vehicle = vehicle.get_bike_inventory()
            for bike in range(0, min(len(bikes_at_station), int(loading_quantity))):
                loading_ids.append(bikes_at_station[bike].id)
            for bike in range(0,min(len(bikes_at_vehicle), int(unloading_quantity))):
                unloading_ids.append(bikes_at_vehicle[bike].id)
 
        print(f"Decision: Go to {station_id}, pickup {len(loading_ids)} bikes, deliver {len(unloading_ids)} bikes")
        return station_id, loading_ids, unloading_ids  
 