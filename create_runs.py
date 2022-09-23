#!/bin/python3
"""
FOMO simulator main program
"""
import copy
import os
import shutil
import json

import settings
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike

import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen

from progress.bar import Bar

import output
import target_state
import matplotlib.pyplot as plt

from helpers import *

RUN_DIRECTORY="experimental_setups"

###############################################################################

# Duration of each simulation run
DURATION = timeInMinutes(hours=24)

# Enter instance definition here.  For numbikes and numstations, enter 'None' to use dataset default
instances = [
    dict(name="Oslo",        url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",        numbikes=None, numstations=None, week=33, day=0, hour=6),
    dict(name="Bergen",      url="https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",      numbikes=None, numstations=None, week=33, day=0, hour=6),
    dict(name="Trondheim",   url="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",   numbikes=None, numstations=None, week=33, day=0, hour=6),
#    dict(name="Oslo-vinter", url="https://data.urbansharing.com/oslovintersykkel.no/trips/v1/",    numbikes=400,  numstations=None, week=7,  day=0, hour=6),
#    dict(name="Edinburgh",   url="https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/", numbikes=200,  numstations=None, week=20, day=0, hour=6),
]

# Enter analysis definition here
analyses = [
    dict(name="do_nothing", target_state="evenly_distributed_target_state", policy="DoNothing",    policyargs="", numvehicles=1),
    dict(name="evenly",     target_state="evenly_distributed_target_state", policy="GreedyPolicy", policyargs="", numvehicles=1),
    dict(name="outflow",    target_state="outflow_target_state",            policy="GreedyPolicy", policyargs="", numvehicles=1),
    dict(name="equalprob",  target_state="equal_prob_target_state",         policy="GreedyPolicy", policyargs="", numvehicles=1),
]        

seeds = list(range(10))

###############################################################################

if __name__ == "__main__":

    starvations = []
    congestions = []

    starvations_stdev = []
    congestions_stdev = []

    if os.path.exists(RUN_DIRECTORY):
        shutil.rmtree(RUN_DIRECTORY)
    os.mkdir(RUN_DIRECTORY)

    n = 0

    for instance in instances:
        starvations.append([])
        congestions.append([])

        starvations_stdev.append([])
        congestions_stdev.append([])

        for analysis in analyses:
            simulations = []

            experimental_setup = dict(run=n, instance=instance, analysis=analysis, seeds=seeds, duration=DURATION)
            with open(f"{RUN_DIRECTORY}/setup_{n:04}.json", "w") as outfile:
                outfile.write(json.dumps(experimental_setup, indent=4))
            n += 1
