#    MOVE TO MAIN FOLDER!!!

###############################################################################

#!/bin/python3
"""
FOMO simulator, create jobs to run on cluster
"""
import os
import shutil
import json

from helpers import *

RUN_DIRECTORY="experimental_setups"
INSTANCE_DIRECTORY="instances"


###############################################################################

# Duration of each simulation run
START_TIME = timeInMinutes(hours=7)
NUM_DAYS = 7
DURATION = timeInMinutes(hours=24*NUM_DAYS)

# Enter instances here
cities = ["Oslo","Bergen","Trondheim","Edinburgh"]
abbrvs = {"Oslo": 'OS',
          "Bergen": 'BG',
          "Trondheim":'TD' ,
          "Edinburgh":'EH'
          }
weeks = {"Oslo": [10,22,31,50],
          "Bergen": [8,25,35,45],
          "Trondheim":[17,21,34,44] ,
          "Edinburgh":[10,22,31,50]
          }
instances = [abbrvs[city]+'_W'+str(week) for city in cities for week in weeks[city]]
#instances = ['TD_W21','TD_W34']


#perform the following analysis for two different policies!!!
analysis = dict(name="num_reps_1_veh",
         target_state="outflow_target_state",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6)

# Enter analysis definition here
analyses = [analysis]

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
