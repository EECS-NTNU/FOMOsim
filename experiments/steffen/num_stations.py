####################################
# set the right working directory #
###################################

import os 
import sys
from pathlib import Path

desired_path = Path(__file__).parents[2]
os.chdir(desired_path)
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

from progress.bar import Bar

import output
import target_state
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.stats import t, norm

from helpers import *
from analyses.steffen.num_sim_replications.helpers import ci_half_length, approximate_num_reps_absolute

###############################################################################

# Duration of each simulation run

START_TIME = timeInMinutes(hours=7)
NUM_DAYS = 7
DURATION = timeInMinutes(hours=24*NUM_DAYS)

#analysis settings
alpha = 0.05
gamma = 0.05 #relative error
gamma_star = gamma/(1-gamma)
beta = 0.25 #absolute error
min_num_seeds = 4
n_max = 60
analysis_type = 'absolute' #relative1,absolute,...

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


INSTANCE_DIRECTORY="instances"

#perform the following analysis for two different policies!!!
analysis = dict(name="outflow",
         target_state="outflow_target_state",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6)

num_stations = []

for instance in instances:
    print("  instance: ", instance)

    tstate = None
    if "target_state" in analysis:
        tstate = getattr(target_state, analysis["target_state"])

    initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + instance, target_state=tstate)
    
    num_stations.append(len(initial_state.stations)) 

results = pd.DataFrame(list(zip(instances,num_stations)),
               columns =['instance', 'num_stations'])
print(results)
    