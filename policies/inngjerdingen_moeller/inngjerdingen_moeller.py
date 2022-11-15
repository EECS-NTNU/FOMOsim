from policies import Policy
import sim
from policies.inngjerdingen_moeller.parameters_MILP import MILP_data
from policies.inngjerdingen_moeller.mathematical_model import run_model

class InngjerdingenMoellerPolicy(Policy):
    def __init__(self, roaming = True, time_horizon=25, tau=5):
        self.roaming = roaming
        self.time_horizon = time_horizon
        self.tau = tau
        super().__init__()

    def get_best_action(self, simul, vehicle):
        data = MILP_data(simul, self.time_horizon, self.tau)
        data.initalize_parameters()
        gurobi_output=run_model(data, self.roaming)
        next_station, bikes_to_pickup, bikes_to_deliver  = self.return_solution(gurobi_output, vehicle)
        
        return sim.Action(
            None,
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
        station_id = None
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name = variable[0]
            indices = variable[1].split(',')
            if name == 'x' and int(indices[3]) == vehicle.id and round(var.x,0) == 1 and int(indices[1]) != vehicle.location.id:
                if int(indices[2]) < first_move_period:
                    first_move_period = int(indices[2])
                    station_id = int(indices[1])
            elif name == 'q_L' and int(indices[2]) == vehicle.id and round(var.x,0) > 0 and int(indices[0]) == vehicle.location.id:
                loading_quantity += var.x
            elif name == 'q_U' and int(indices[2]) == vehicle.id and round(var.x,0) > 0 and int(indices[0]) == vehicle.location.id:
                unloading_quantity += var.x
        
        for bike in range(0, int(loading_quantity)):
            loading_ids.append(vehicle.location.bikes[bike])
        for bike in range(0,unloading_quantity):
            unloading_ids.append(vehicle.location.bikes[bike])

        return station_id, loading_ids, unloading_ids   

