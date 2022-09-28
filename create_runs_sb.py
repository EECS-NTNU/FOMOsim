#!/bin/python3
"""
FOMO simulator, create jobs to run on cluster
"""
import os
import shutil
import json
import numpy as np
import copy

from helpers import *

RUN_DIRECTORY="experimental_setups"




###############################################################################

# Duration of each simulation run
DURATION = timeInMinutes(hours=24*7)

# Enter instance definition here.  For numbikes and numstations, enter 'None' to use dataset default

num_weeks = 6
#weeks = [round(x) for x in np.linspace(start=4,stop=48,num=num_weeks)]   #little activity in week 48
weeks = [10,20,33,40]

instance_base_setups = [
dict(name="Oslo_W33",        city = "Oslo",     url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",        numbikes=2000, numstations=None, week=33, day=0, hour=6),
dict(name="Bergen_W33",      city = "Bergen",   url="https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",      numbikes=1000, numstations=None, week=33, day=0, hour=6),
dict(name="Trondheim_W33",   city = "Trondheim",url="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",   numbikes=1000,numstations=None, week=33, day=0, hour=6)
]

instances = []
for week in weeks:
     for instance in instance_base_setups:
         instance_copy=copy.deepcopy(instance)
         instance_copy['name']=instance_copy['city']+'_W'+ str(week)
         instance_copy['week'] = week
         instances.append(instance_copy)

ts_map = {
    #"EDTS":"evenly_distributed_target_state",
    "OFTS":"outflow_target_state",
    "EQTS":"equal_prob_target_state"
        }
policy_map = {
    #abbreviation:name_of_policy
    "GRD":"GreedyPolicy" #no need to find all combinations
    }

all_weights = get_criticality_weights2(4)
policyargs={}
number_of_vehicles = [1,2]

do_nothing_analysis = dict(
    name='do_nothing',
    target_state="evenly_distributed_target_state",
    policy='DoNothing',
    numvehicles=1,
    policyargs={}
    )

analyses = [do_nothing_analysis]   #reference_case
for ts_abbr,ts in ts_map.items():
    for pol_abbr, pol in policy_map.items():
        for nv in number_of_vehicles:
            for crit_weight in all_weights:
                analyses.append(dict(
                    name=ts_abbr+'_'+pol_abbr+'_'+'V'+str(nv)+'_W'+str(crit_weight),
                    target_state=ts,
                    policy=pol,
                    numvehicles=nv,
                    policyargs={'crit_weights':crit_weight}
                    ))

# Enter analysis definition here
# analyses = [
#     dict(name="do_nothing", target_state="evenly_distributed_target_state", policy="DoNothing",    policykwargs={}, numvehicles=1 ),
#     dict(name="evenly",     target_state="evenly_distributed_target_state", policy="GreedyPolicy", policykwargs={}, numvehicles=1),
#     dict(name="outflow",    target_state="outflow_target_state",            policy="GreedyPolicy", policykwargs={}, numvehicles=1),
#     dict(name="equalprob",  target_state="equal_prob_target_state",         policy="GreedyPolicy", policykwargs={}, numvehicles=1),
# ]        

seeds = list(range(20))

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

    f = open('analyses.csv','w')
    delim=';'
    f.write(delim.join(list(analyses[0].keys())))
    f.write("\n")
    for analysis in analyses:
        f.write(analysis['name'] + ";")
        f.write(analysis['target_state'] + ";")
        f.write(analysis['policy'] + ";")
        f.write(str(analysis['numvehicles']) + ";")
        for key,value in analysis['policyargs'].items():
            f.write(key + ";")
            if isinstance(value, list):
                for i in value:
                    f.write(str(i) + ";")
            else:
                f.write(str(value) + ";")
        f.write("\n")
    f.close()
