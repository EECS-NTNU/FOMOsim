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

from create_runs_base_settings import *

from helpers import *

RUN_DIRECTORY="experimental_setups"

some_runs_already_performed = True

finished_tasks = []
if some_runs_already_performed:
    finished_tasks = []
    with open('completed_tasks3.csv', newline='') as f:
        for row in csv.reader(f):
            finished_tasks.append(int(row[0]))

print('number of finished tasks: ', str(len(finished_tasks)))

###############################################################################

# Duration of each simulation run
#NUM_DAYS = 7
#DURATION = timeInMinutes(hours=24*NUM_DAYS)

# Enter instance definition here.  

cities = ["Oslo","Bergen","Trondheim"] #CITIES
abbrvs = ABBRVS
weeks = WEEKS
instances = [abbrvs[city]+'_W'+str(week) for city in cities for week in weeks[city]]

num_seeds = NUM_SEEDS

# ANALYSES

do_nothing_analysis = dict(name="do_nothing",
         numvehicles=0,
         day=0,
         hour=6)
         
analyses = [do_nothing_analysis]   #reference_case


ts_map = {
    #"EVEN":"EvenlyDistributedTargetState",
    #"HALF":"HalfCapacityTargetState",
    "EQUAL":"USTargetState"
    #"OFTS":"OutflowTargetState",
    #"EQTS":"EqualProbTargetState"
        }
policy_map = {
    #abbreviation:name_of_policy
    "GRD":"GreedyPolicy", #no need to find all combinations
    #"GHB":"policies.gleditsch_hagen.GleditschHagenPolicy"
    }

all_weights = get_criticality_weights2(4) #75 combinations (my calculations gave 74, check the difference)

policyargs={}
number_of_vehicles = [1,2]

for ts_abbr,ts in ts_map.items():
    for pol_abbr, pol in policy_map.items():
        for nv in number_of_vehicles:
            for crit_weight in all_weights:
                analyses.append(dict(
                    name=ts_abbr+'_'+pol_abbr+'_'+'V'+str(nv)+'_W'+str(crit_weight),
                    target_state=ts,
                    policy=pol,
                    numvehicles=nv,
                    day = 0,
                    hour = 6,
                    policyargs={'crit_weights':crit_weight,'service_hours':[8,16]}
                    ))

print('max number of analyses: ', len(analyses)*len(instances))



#seeds = list(range(NUM_SEEDS))

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

    # f = open('analyses.csv','w')
    # delim=';'
    # f.write(delim.join(list(analyses[0].keys())))
    # f.write("\n")
    # for analysis in analyses:
    #     f.write(analysis['name'] + ";")
    #     f.write(str(analysis['numvehicles']) + ";")
    #     if analysis['numvehicles']>0:
    #         f.write(analysis['target_state'] + ";")
    #         f.write(analysis['policy'] + ";")
    #         for key,value in analysis['policyargs'].items():
    #             f.write(key + ";")
    #             if isinstance(value, list):
    #                 for i in value:
    #                     f.write(str(i) + ";")
    #             else:
    #                 f.write(str(value) + ";")
    #     f.write("\n")
    # f.close()
