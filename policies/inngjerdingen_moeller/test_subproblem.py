from parameters_MILP import MILP_data
from mathematical_model import run_model
from inngjerdingen_moeller import InngjerdingenMoellerPolicy

import sim
import policies
import demand
from init_state.wrapper import read_initial_state
import target_state
from helpers import timeInMinutes
from visualize_subproblem import Visualizer


if __name__ == "__main__":
# ------------ TESTING DATA MANUALLY ---------------
        #filename = "instances/EH_W31"
        filename = "instances/TD_W34"
        #filename = "instances/OS_W31"

        START_DAY = 0
        START_TIME = timeInMinutes(hours=7)
        DURATION = timeInMinutes(hours=1)

        state1 = read_initial_state(filename)
        state1.set_seed(1)
        
        # tstate = target_state.EvenlyDistributedTargetState()
        # tstate = target_state.OutflowTargetState()
        # tstate = target_state.EqualProbTargetState()
        # tstate = target_state.USTargetState()
        tstate = target_state.HalfCapacityTargetState()
        tstate.update_target_state(state1,START_DAY,7)
        
        policy = policies.GreedyPolicy()
        
        state1.set_vehicles([policy, policy])
        
        
        dmand = demand.Demand()
        dmand.update_demands(state1,START_DAY,START_TIME)

        simul1 = sim.Simulator(
                initial_state = state1,
                target_state = tstate,
                demand = dmand,
                start_time = START_TIME,
                duration = DURATION,
                verbose = True,
        )
     
        d=MILP_data(simul1, 20, 5)
        d.initalize_parameters()
        print("TESTING COMPLETE")
        m=run_model(d, True)
        v=Visualizer(m,d)
        v.visualize_route()
# ----------------------------------------------------
