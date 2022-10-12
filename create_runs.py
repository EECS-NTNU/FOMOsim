#!/bin/python3
"""
FOMO simulator, create jobs to run on cluster
"""
import os
import shutil
import json

from helpers import *

RUN_DIRECTORY="experimental_setups"

###############################################################################

# Duration of each simulation run
DURATION = timeInMinutes(hours=24)

# Enter instances here
instances = [ "Oslo", "Bergen", "Trondheim", "Edinburgh" ]

# Enter analysis definition here
analyses = [

    dict(name="do_nothing",
         numvehicles=0,
         day=0,
         hour=6),

    #flat strategy
    dict(name="evenly",
         target_state="evenly_distributed_target_state",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0.25,0.25,0.25,0.25]},
         numvehicles=1,
         day=0,
         hour=6),    

    #deviation_from_target_state
    dict(name="outflow",
         target_state="outflow_target_state",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=1,
         day=0,
         hour=6),     

    dict(name="equalprob",
         target_state="equal_prob_target_state",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6),

]

seeds = list(range(10))

###############################################################################

if __name__ == "__main__":

    if os.path.exists(RUN_DIRECTORY):
        shutil.rmtree(RUN_DIRECTORY)
    os.mkdir(RUN_DIRECTORY)

    n = 0

    for instance in instances:
        for analysis in analyses:
            simulations = []

            experimental_setup = dict(run=n, instance=instance, analysis=analysis, seeds=seeds, duration=DURATION)
            with open(f"{RUN_DIRECTORY}/setup_{n:04}.json", "w") as outfile:
                outfile.write(json.dumps(experimental_setup, indent=4))
            n += 1
