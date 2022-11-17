from parameters_MILP import MILP_data
from mathematical_model import run_model
from inngjerdingen_moeller import InngjerdingenMoellerPolicy

import sim
import demand
from init_state.wrapper import read_initial_state
import target_state
from helpers import timeInMinutes
from visualize_subproblem import Visualizer
import json


if __name__ == "__main__":
# ------------ TESTING DATA MANUALLY ---------------
        filename = "instances/EH_W31"
        #filename = "instances/TD_W34" 
        #filename = "instances/OS_W31"

        START_DAY = 0 #0 -> monday ,days other than 0 results in target inventory = 0 for all stations
        START_HOUR = 8 #8 -> 08:00 am
        START_TIME = timeInMinutes(hours=START_HOUR)
        DURATION = timeInMinutes(hours=1)

        state1 = read_initial_state(filename)
        state1.set_seed(1) 
        
        # tstate = target_state.EvenlyDistributedTargetState()
        # tstate = target_state.OutflowTargetState()
        # tstate = target_state.EqualProbTargetState()
        # tstate = target_state.USTargetState()
        tstate = target_state.HalfCapacityTargetState()
        
        tstate.update_target_state(state1,START_DAY,START_HOUR)
        
        policy = InngjerdingenMoellerPolicy()
        
        state1.set_vehicles([policy]) #number of policy objects in list determines number of vehicles
        state1.vehicles[0].location = state1.locations[1] #initial vehicle position
        
        dmand = demand.Demand()
        dmand.update_demands(state1, START_DAY, START_HOUR)

        simul1 = sim.Simulator(
                initial_state = state1,
                target_state = tstate,
                demand = dmand,
                start_time = START_TIME,
                duration = DURATION,
                verbose = True,
        )
     
        d=MILP_data(simul1, 15, 5) #input parameters determine time horizon and length of time period (tau)
        d.initalize_parameters()
        print("TESTING COMPLETE")
        m=run_model(d, True) #True if run model with roaming
        v=Visualizer(m,d)
        v.visualize_route() 
# ----------------------------------------------------