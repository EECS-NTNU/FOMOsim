from policies import Policy
import sim
from policies.inngjerdingen_moeller.parameters_MILP import MILP_data
from policies.inngjerdingen_moeller.mathematical_model import run_model
from policies.inngjerdingen_moeller.visualize_subproblem import Visualizer


class InngjerdingenMoellerPolicy(Policy):
    def __init__(self, roaming = True, time_horizon=25, tau=5, weights=None):
        self.roaming = roaming
        self.time_horizon = time_horizon
        self.tau = tau
        self.weights = weights
        super().__init__()

    def get_best_action(self, simul, vehicle):
        data = MILP_data(simul, self.time_horizon, self.tau, self.weights)
        data.initalize_parameters()
        gurobi_output = run_model(data, self.roaming)
        next_station, bikes_to_pickup, bikes_to_deliver  = self.return_solution(gurobi_output, vehicle)
        # gurobi_output.printAttr("X")
        # v=Visualizer(gurobi_output,data) 
        # v.visualize_route()

################################################################################
        #Compare without roaming
        # gurobi_output_no_roaming = run_model(data, False)
        # next_station2, bikes_to_pickup2, bikes_to_deliver2  = self.return_solution(gurobi_output_no_roaming, vehicle)
        # if next_station2 != next_station: 
        #     simul.metrics.add_aggregate_metric(simul, "different_station_choice", 1)

        # if len(bikes_to_pickup2) != len(bikes_to_pickup): #only care about length of bicycle list
        #     simul.metrics.add_aggregate_metric(simul, "different_pickup_quantity", 1)
        # elif len(bikes_to_deliver2) != len(bikes_to_deliver):
        #     simul.metrics.add_aggregate_metric(simul, "different_deliver_quantity", 1)
        
        # if next_station2 != next_station and (len(bikes_to_pickup2) != len(bikes_to_pickup) or len(bikes_to_deliver2) != len(bikes_to_deliver)):
        #     simul.metrics.add_aggregate_metric(simul, "overlap", 1)

        # if next_station2 == next_station and len(bikes_to_pickup2) == len(bikes_to_pickup) and len(bikes_to_deliver2) == len(bikes_to_deliver):
        #     simul.metrics.add_aggregate_metric(simul, "same_choice", 1)

        # simul.metrics.add_aggregate_metric(simul, "number_of_subproblems", 1) 
        
################################################################################
        
        return sim.Action(
            [],               # batteries to swap
            bikes_to_pickup, #list of bike id's
            bikes_to_deliver, #list of bike id's
            next_station, #id
        )   

        
    def return_solution(self, gurobi_output, vehicle):
        first_move_period = 1000
        loading_quantity = 0
        unloading_quantity = 0
        loading_ids = []
        unloading_ids = []
        station_id = vehicle.location.location_id
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name = variable[0]
            indices = variable[1].split(',')
            if name == 'x' and int(indices[3]) == vehicle.vehicle_id and round(var.x,0) == 1 and int(indices[1]) != vehicle.location.location_id and int(indices[1]) != -1:
                if int(indices[2]) < first_move_period:
                    first_move_period = int(indices[2])
                    station_id = int(indices[1])

        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name = variable[0]
            indices = variable[1].split(',')
            if name == 'q_L' and int(indices[2]) == vehicle.vehicle_id and round(var.x,0) > 0 and int(indices[0]) == vehicle.location.location_id and int(indices[1]) <= first_move_period:
                loading_quantity += var.x
            elif name == 'q_U' and int(indices[2]) == vehicle.vehicle_id and round(var.x,0) > 0 and int(indices[0]) == vehicle.location.location_id and int(indices[1]) <= first_move_period:
                unloading_quantity += var.x
                
        if not (loading_quantity == 0 and unloading_quantity == 0):
            bikes_at_station = list(vehicle.location.bikes.values()) #creates list of bike objects
            bikes_at_vehicle = vehicle.get_bike_inventory() 
            for bike in range(0, min(len(bikes_at_station), int(loading_quantity))): 
                loading_ids.append(bikes_at_station[bike].id)
            for bike in range(0,min(len(bikes_at_vehicle), int(unloading_quantity))):
                unloading_ids.append(bikes_at_vehicle[bike].id)

        return station_id, loading_ids, unloading_ids  
