####################################
# set the right working directory #
###################################

import os 
import sys
from pathlib import Path

path = Path(__file__).parents[2]
os.chdir(path)
#print(os. getcwd())

sys.path.insert(0, '') #make sure the modules are found in the new working directory

###############################################################################

#!/bin/python3
"""
FOMO simulator main program
"""
import copy

import settings
import sim
import init_state
import init_state.cityBike

import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import demand
from progress.bar import Bar

import output
import target_state
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import t, norm

from helpers import *
from analyses.steffen.num_sim_replications.helpers import *

###############################################################################

# Duration of each simulation run

START_TIME = timeInMinutes(hours=7)
NUM_DAYS = 7
DURATION = timeInMinutes(hours=24*NUM_DAYS)
instance = "BG_W35"
INSTANCE_DIRECTORY="instances"

analysis = dict(name="equalprob",
         target_state="HalfCapacityTargetState",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6)



if __name__ == "__main__":


    tstate = None
    if "target_state" in analysis:
        tstate = getattr(target_state, analysis["target_state"])()

    initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + instance)
    
    if analysis["numvehicles"] > 0:
        policyargs = analysis["policyargs"]
        policy = getattr(policies, analysis["policy"])(**policyargs)
        initial_state.set_vehicles([policy]*analysis["numvehicles"])

    num_seeds = 5
    seeds = list(range(num_seeds))

    lost_trips_starv = []
    lost_trips_cong = []

    lost_trips_starv_stdev = []
    lost_trips_cong_stdev = []

    simulations = []

    for seed in seeds:
        print("      seed: ", seed)

        state_copy = copy.deepcopy(initial_state)
        state_copy.set_seed(seed)

        simul = sim.Simulator(
            initial_state = state_copy,
            target_state = tstate,
            demand = demand.Demand(),
            start_time = timeInMinutes(days=analysis["day"], hours=analysis["hour"]),
            duration = DURATION,
            verbose = True,
        )
        
        simul.run()
        
        scale = 100 / simul.metrics.get_aggregate_value("trips")
        perc_lost_trips = scale*(simul.metrics.get_aggregate_value("starvation")+
                            simul.metrics.get_aggregate_value("congestion"))
        simul.metrics.add_metric(simul,'perc_lost_trips',perc_lost_trips)
        print(simul.metrics.metrics['perc_lost_trips'])
        simulations.append(simul)
        
        #scale = 100 / simul.metrics.get_aggregate_value("trips")
        #starvations.append(scale*simul.metrics.get_aggregate_value("starvation"))
        #congestions.append(scale*simul.metrics.get_aggregate_value("congestion"))

    metric = sim.Metric.merge_metrics([simul.metrics for simul in simulations])  #merging the seeds into something overordnet
    print(metric.metrics['perc_lost_trips'])

    print(metric.metrics["perc_lost_trips"])
    print(metric.get_aggregate_value("perc_lost_trips"))

    scale = 100 / metric.get_aggregate_value("trips")

    lost_trips_starv.append(scale * metric.get_aggregate_value("starvation"))
    lost_trips_cong.append(scale * metric.get_aggregate_value("congestion"))

    lost_trips_starv_stdev.append(scale * metric.get_aggregate_value("starvation_stdev"))
    lost_trips_cong_stdev.append(scale * metric.get_aggregate_value("congestion_stdev"))

   # lost_trips_stdev[-1].append(np.sqrt((scale * metric.get_aggregate_value("starvation_stdev"))**2)

    ###############################################################################



