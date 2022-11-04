#JUST MOVE TO THE MAIN FOLDER!!

# import os 
# import sys
# from pathlib import Path

# path = Path(__file__).parents[3]
# os.chdir(path)
# #print(os. getcwd())

# sys.path.insert(0, '') #make sure the modules are found in the new working directory

###############################################################################


#!/bin/python3
"""
FOMO simulator, create jobs to run on cluster
"""
import os
import shutil
import json
import numpy as np
import copy
import csv

from helpers import *
from create_runs_base_settings import *

RUN_DIRECTORY="experimental_setups"

some_runs_already_performed = True

finished_tasks = []
if some_runs_already_performed:
    finished_tasks = []
    with open('completed_tasks.csv', newline='') as f:
        for row in csv.reader(f):
            finished_tasks.append(int(row[0]))


###############################################################################

# Duration of each simulation run
#NUM_DAYS = 7
#DURATION = timeInMinutes(hours=24*NUM_DAYS)

# Enter instance definition here.  

cities = [  "Oslo",
            "Bergen",
            "Trondheim",
            #"Edinburgh",
            ]

#cities = CITIES
abbrvs = ABBRVS
weeks = WEEKS

instances = [abbrvs[city]+'_W'+str(week) for city in cities for week in weeks[city]]

num_seeds = NUM_SEEDS

# ANALYSES

do_nothing_analysis = dict(
    name="do_nothing",
    #target_state = None,
    numvehicles=0,
    day=0,
    hour=6)
         
analyses = [do_nothing_analysis]   #reference_case

number_of_vehicles = [1,2]

all_weights = [ [0,0,0,1],
                [0.1,0.2,0.3,0.4],
                ]

ts_map = {
    "EVEN":"EvenlyDistributedTargetState",
    "HALF":"HalfCapacityTargetState",
    "EQUAL":"USTargetState",
    #
    #"OF":"OutflowTargetState",
    #"EQ":"EqualProbTargetState",
    }
policy_map = {
    #abbreviation:name_of_policy
    "GRD":"GreedyPolicy", #no need to find all combinations
    #"GHB":"policies.gleditsch_hagen.GleditschHagenPolicy"
    }

policyargs={}

for ts_abbr,ts in ts_map.items():
    for pol_abbr, pol in policy_map.items():
        for nv in number_of_vehicles:
            for crit_weight in all_weights:
                analyses.append(dict(
                    name=ts_abbr+'_'+pol_abbr+'_'+'V'+str(nv)+'_W'+str(crit_weight)+'_8to6',
                    target_state=ts,
                    policy=pol,
                    numvehicles=nv,
                    day = 0,
                    hour = 6,
                    policyargs={'crit_weights':crit_weight,'service_hours':[8,16]}
                    ))

print(len(analyses))
print(len(instances))
print(len(analyses)*len(instances))


###############################################################################

if __name__ == "__main__":

    if os.path.exists(RUN_DIRECTORY):
        shutil.rmtree(RUN_DIRECTORY)
    os.mkdir(RUN_DIRECTORY)

    n = 0

    for instance in instances:
        for analysis in analyses:
            if (n not in finished_tasks):
                simulations = []
                experimental_setup = dict(run=n, instance=instance, analysis=analysis, seeds=list(range(num_seeds[instance])), duration=DURATION)
                with open(f"{RUN_DIRECTORY}/setup_{n:04}.json", "w") as outfile:
                    outfile.write(json.dumps(experimental_setup, indent=4))
            n += 1