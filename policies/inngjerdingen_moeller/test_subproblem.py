from parameters_MILP import MILP_data
from mathematical_model import run_model

import sim
import policies

from init_state.wrapper import read_initial_state
from target_state import equal_prob_target_state, evenly_distributed_target_state
from helpers import timeInMinutes


if __name__ == "__main__":
# ------------ TESTING DATA MANUALLY ---------------
        #filename = "instances/EH_W31"
        filename = "instances/TD_W34"
        #filename = "instances/OS_W31"

        #tstate = equal_prob_target_state
        tstate = evenly_distributed_target_state
        
        state1 = read_initial_state(filename, tstate)
        policy = policies.GreedyPolicy()
        
        state1.set_vehicles([policy])
        
        simul1 = sim.Simulator(
                initial_state = state1,
                start_time = timeInMinutes(hours=7), #hours=7, hours=11 or hours=15
                duration = timeInMinutes(hours=1),
                verbose = True,
        )

        d=MILP_data(simul1, 25, 5)
        d.initalize_parameters()
        print("TESTING COMPLETE")
        run_model(d, True)
# ----------------------------------------------------
