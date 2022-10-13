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

import numpy as np
from scipy.stats import t, norm

from helpers import *
from analyses.steffen.num_sim_replications.helpers import *

###############################################################################

# Duration of each simulation run

START_TIME = timeInMinutes(hours=7)
NUM_DAYS = 7
DURATION = timeInMinutes(hours=24*NUM_DAYS)
instance = "BG_w35"
INSTANCE_DIRECTORY="instances"

analysis = dict(name="equalprob",
         target_state="equal_prob_target_state",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6)



if __name__ == "__main__":


    tstate = None
    if "target_state" in analysis:
        tstate = getattr(target_state, analysis["target_state"])

    initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + instance, target_state=tstate)
    
    if analysis["numvehicles"] > 0:
        policyargs = analysis["policyargs"]
        policy = getattr(policies, analysis["policy"])(**policyargs)
        initial_state.set_vehicles([policy]*analysis["numvehicles"])

    num_seeds = 100
    seeds = list(range(num_seeds))

    starvations = []
    congestions = []

    starvations_stdev = []
    congestions_stdev = []

    for seed in seeds:
        print("      seed: ", seed)

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

    ###############################################################################


    alpha = 0.05
    relative_error = 0.05 #gamma
    absolute_error = 1 #beta   #note that the original values are between 0 and 100

    m = 100

    print('---------------------')
    print('approach 1 (brute force relative)')
    print('---------------------')

    for key,random_sample in dict(congestions=congestions,starvations=starvations).items():
        
        print(key)
        print('---------------------')

        
        values = []
        for mm in range(m):
            nn = approximate_num_reps_relative1(random_sample[0:mm],relative_error,alpha,5)   
            values.append(nn)
        print('n:')
        print(values)

    print('---------------------')
    print('approach 2 (relative build up)')
    print('---------------------')

    for key,random_sample in dict(congestions=congestions,starvations=starvations).items():
        
        print(key)
        print('---------------------')

        values = []
        for mm in range(m):
            nn = approximate_num_reps_relative2(random_sample[0:mm],relative_error,alpha,5)   
            values.append(nn)
        print('n:')
        print(values)

    print('---------------------')
    print('approach 3 (sequential)')      #THIS ONE SEEMS TO BE MOST ACCURATE!
    print('---------------------')

    for key,random_sample in dict(congestions=congestions,starvations=starvations).items():
        
        print(key)
        nn = approximate_num_reps_relative2(random_sample[0:mm],relative_error,alpha,5)  
        print('n=', nn)
        print('---------------------')

        values = []
        for mm in range(m):
            
            values.append(nn)
        print('n:')
        print(values)

    print('---------------------')
    print('approach 4 (absolute build up)')
    print('---------------------')

    override = True
    if override == True:
        absolute_error = 0.3  #overriding previously

    for key,random_sample in dict(congestions=congestions,starvations=starvations).items():
        
        print(key)
        print('---------------------')

        values = []
        for mm in range(m):
            nn = approximate_num_reps_absolute(random_sample[0:mm],absolute_error,alpha,5)   
            values.append(nn)
        print('n:')
        print(values)


    #INTERESTING RESULTS HERE!!! 
    # if we only look at starvations, then only 5 seeds are needed to get the confidence, however for congestions, 9 seeds are needed

    m = 20
    random_sample = congestions
    sample_std = np.std(random_sample[0:m])
    sample_mean = np.mean(random_sample[0:m])
    print(sample_mean)
    


