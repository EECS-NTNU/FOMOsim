####################################
# set the right working directory #
###################################

import os 
import sys
from pathlib import Path

path = Path(__file__).parents[3]
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

from progress.bar import Bar

import output
import target_state
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.stats import t, norm

from helpers import *
from analyses.steffen.num_sim_replications.helpers import ci_half_length

###############################################################################

# Duration of each simulation run

START_TIME = timeInMinutes(hours=7)
NUM_DAYS = 7
DURATION = timeInMinutes(hours=24*NUM_DAYS)

#analysis settings
alpha = 0.05
gamma = 0.1 #relative error
min_num_seeds = 4

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
#instances = ['TD_W21']

INSTANCE_DIRECTORY="instances"

analysis = dict(name="outflow",
         target_state="outflow_target_state",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6)

if __name__ == "__main__":

    n_starvations = []
    n_congestions = []

    for instance in instances:
        print("  instance: ", instance)


        tstate = None
        if "target_state" in analysis:
            tstate = getattr(target_state, analysis["target_state"])

        initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + instance, target_state=tstate)

        if analysis["numvehicles"] > 0:
            policyargs = analysis["policyargs"]
            policy = getattr(policies, analysis["policy"])(**policyargs)
            initial_state.set_vehicles([policy]*analysis["numvehicles"])

        
        

        #FIRST DO THE MINIMUM NUMBER OF SEEDS
        
        starvations = []
        congestions = []
        for seed in range(min_num_seeds):

            state_copy = copy.deepcopy(initial_state)
            state_copy.set_seed(seed)

            simul = sim.Simulator(
                initial_state = state_copy,
                start_time = timeInMinutes(days=analysis["day"], hours=analysis["hour"]),
                duration = DURATION,
                verbose = True,
            )
            
            simul.run()

            scale = 100 / simul.metrics.get_aggregate_value("trips")
            starvations.append(scale*simul.metrics.get_aggregate_value("starvation"))
            congestions.append(scale*simul.metrics.get_aggregate_value("congestion"))

            mean_starv = np.mean(starvations)
            mean_cong = np.mean(congestions)
            std_starv = np.std(starvations)
            std_cong = np.std(congestions)

        #THEN, START ON THE ITERATIVE PROCEDURE

        n = min_num_seeds
        
        starv_finished=False
        cong_finished=False


        finished = False
        while not finished:

            n += 1

            state_copy = copy.deepcopy(initial_state)
            state_copy.set_seed(n)
            simul = sim.Simulator(
                initial_state = state_copy,
                start_time = timeInMinutes(days=analysis["day"], hours=analysis["hour"]),
                duration = DURATION,
                verbose = True,
            )
            simul.run()

            scale = 100 / simul.metrics.get_aggregate_value("trips")
            starv = scale*simul.metrics.get_aggregate_value("starvation")
            cong = scale*simul.metrics.get_aggregate_value("congestion")
            starvations.append(starv)
            congestions.append(cong)

            mean_starv = np.mean(starvations)
            mean_cong = np.mean(congestions)
            std_starv = np.std(starvations)
            std_cong = np.std(congestions)

            if starv_finished==False:
                if (ci_half_length(n,alpha,std_starv) / np.abs(mean_starv) <= gamma/(1-gamma)):
                    starv_finished = True
                    n_starvations.append(n) 

            if cong_finished == False:
                if (ci_half_length(n,alpha,std_cong) / np.abs(mean_cong) <= gamma/(1-gamma)):
                    cong_finished = True
                    n_congestions.append(n)
            
            if starv_finished and cong_finished:
                finished = True

            if n > 60:
                finished = True

    results = pd.DataFrame(list(zip(instances, n_starvations,n_congestions)),
               columns =['instance', 'n_starv','n_cong'])
    print(results)
    results.to_csv('num_rep_analysis.csv')
    directory = 'analyses/steffen/num_sim_replications'
    results.to_csv(directory+'/num_rep_analysis.csv')

            
            

            






