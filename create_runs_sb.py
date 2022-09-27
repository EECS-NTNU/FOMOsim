#!/bin/python3
"""
FOMO simulator, create jobs to run on cluster
"""
import os
import shutil
import json
import numpy as np

from helpers import *

RUN_DIRECTORY="experimental_setups"




###############################################################################

# Duration of each simulation run
DURATION = timeInMinutes(hours=24)

# Enter instance definition here.  For numbikes and numstations, enter 'None' to use dataset default
instances = [
    dict(name="Oslo",        url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",        numbikes=2000, numstations=None, week=33, day=0, hour=6),
#    dict(name="Bergen",      url="https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",      numbikes=1000, numstations=None, week=33, day=0, hour=6),
#    dict(name="Trondheim",   url="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",   numbikes=1000, numstations=None, week=33, day=0, hour=6),
#    dict(name="Oslo-vinter", url="https://data.urbansharing.com/oslovintersykkel.no/trips/v1/",    numbikes=400,  numstations=None, week=7,  day=0, hour=6),
#    dict(name="Edinburgh",   url="https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/", numbikes=200,  numstations=None, week=20, day=0, hour=6),
]


ts_map = {
    #"EDTS":"evenly_distributed_target_state",
    "OFTS":"outflow_target_state",
    #"EQTS":"equal_prob_target_state"
        }
policy_map = {
    #abbreviation:name_of_policy
    #"DN:""DoNothing",
    "GRD":"GreedyPolicy"
    }
delta = 0.1
w1_range= w2_range= w3_range= w4_range = [0,1]  # time_to_violation, net_demand, driving_time, deviation_target_state
all_weights = get_criticality_weights(delta, w1_range, w2_range,w3_range,w4_range)
policyargs={}
number_of_vehicles = [2]

analyses = []
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
    for key,value in analysis['policyargs'].items():
        f.write(key + ";")
        if isinstance(value, list):
            for i in value:
                f.write(str(i) + ";")
        else:
            f.write(str(value) + ";")
    f.write("\n")
f.close()
