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

# Enter instance definition here.  For numbikes and numstations, enter 'None' to use dataset default
instances = [
    dict(name="Oslo",        url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",        numbikes=2000, numstations=None, week=33, day=0, hour=6),
    dict(name="Bergen",      url="https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",      numbikes=1000, numstations=None, week=33, day=0, hour=6),
    dict(name="Trondheim",   url="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",   numbikes=1000, numstations=None, week=33, day=0, hour=6),
#    dict(name="Oslo-vinter", url="https://data.urbansharing.com/oslovintersykkel.no/trips/v1/",    numbikes=400,  numstations=None, week=7,  day=0, hour=6),
#    dict(name="Edinburgh",   url="https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/", numbikes=200,  numstations=None, week=20, day=0, hour=6),
]

# Enter analysis definition here
analyses = [
    dict(name="do_nothing", target_state="evenly_distributed_target_state", policy="DoNothing",    policyargs={}, numvehicles=1),
    dict(name="evenly",     target_state="evenly_distributed_target_state", policy="GreedyPolicy", policyargs={'crit_weights':[0.25,0.25,0.25,0.25]}, numvehicles=1),    #flat strategy
    dict(name="outflow",    target_state="outflow_target_state",            policy="GreedyPolicy", policyargs={'crit_weights':[0,0,0,1]}, numvehicles=1),     #deviation_from_target_state
    dict(name="equalprob",  target_state="equal_prob_target_state",         policy="GreedyPolicy", policyargs={}, numvehicles=1),
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
