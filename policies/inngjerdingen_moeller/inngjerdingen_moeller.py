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
        gurobi_output=run_model(data, self.roaming)
        bikes_to_pickup, bikes_to_deliver, next_station = self.return_solution(gurobi_output, vehicle)
        
        return sim.Action(
            None,
            bikes_to_pickup, #list of bike id's
            bikes_to_deliver, #list of bike id's
            next_station, #id
        )   

        
    def return_solution(self, gurobi_output, vehicle):
        first_move_period = 1000
        loading = 0
        unloading = 0
        station_id = None
        for var in gurobi_output.getVars():
            variable = var.varName.strip("]").split("[")
            name = variable[0]
            indices = variable[1].split(',')
            if name == 'x' and indices[3] == vehicle.id and round(var.x,0)==1 and indices[1] != vehicle.location.location_id:
                if indices[2] < first_move_period:
                    first_move_period = indices[2]
                    station_id = indices[1]
            if name == 'q_L' and indices[2] == 1 

        return station_id, loading, unloading   

