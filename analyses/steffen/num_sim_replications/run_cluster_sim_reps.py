
###############################################################################

#!/bin/python3
"""
FOMO simulator main program
"""
import copy

import sys
from threading import Timer

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

import json


###############################################################################

# Duration of each simulation run



#analysis settings
alpha = 0.05
gamma = 0.05 #relative error
gamma_star = gamma/(1-gamma)
beta = 0.25 #absolute error
min_num_seeds = 4
n_max = 60
analysis_type = 'absolute_iterate' #relative1,absolute,absolute_iterate

INSTANCE_DIRECTORY="instances"

from math import ceil
def round_up_to_multiple_of_n(number, n):
    return ceil((number+0.05) / n)* n
    

if __name__ == "__main__":

    for filename in sys.argv[1:]:
            with open(filename, "r") as infile:
                print("Running file", filename) #filename='experimental_setups/setup_0063.json'

                time_start = datetime.now() 

                experimental_setup = json.load(infile)
                analysis = experimental_setup["analysis"]
                DURATION = experimental_setup["duration"]

                tstate = None
                if "target_state" in experimental_setup["analysis"]:
                    tstate = getattr(target_state, experimental_setup["analysis"]["target_state"])

                initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + experimental_setup["instance"], target_state=tstate)

                if experimental_setup["analysis"]["numvehicles"] > 0:
                    policyargs = experimental_setup["analysis"]["policyargs"]
                    policy = getattr(policies, experimental_setup["analysis"]["policy"])(**policyargs)
                    initial_state.set_vehicles([policy]*experimental_setup["analysis"]["numvehicles"])

                #simulations = []
                #for seed in experimental_setup["seeds"]:

                results = None

                #for instance in instances:
                
                if analysis_type == 'relative1':
                
                    print('to do')

                elif analysis_type == 'absolute':

                    print('to do')
                    
                elif analysis_type == 'absolute_iterate':

                    starvations = []
                    congestions = []
                    violations = []
                    trips = []

                    for seed in range(min_num_seeds):
                        #seed = 0

                        state_copy = copy.deepcopy(initial_state)
                        state_copy.set_seed(seed)

                        simul = sim.Simulator(
                            initial_state = state_copy,
                            start_time = timeInMinutes(days=analysis["day"], hours=analysis["hour"]),
                            duration = DURATION,
                            verbose = True,
                        )
                        
                        simul.run()

                        trip = simul.metrics.get_aggregate_value("trips")
                        starvation = simul.metrics.get_aggregate_value("starvation")
                        congestion = simul.metrics.get_aggregate_value("congestion")
                        violation = starvation+congestion
                        
                        scale = 100 / trip
                        starvations.append(scale*starvation)
                        congestions.append(scale*congestion)
                        violations.append(scale*violation)


                    #THEN, START ON THE ITERATIVE PROCEDURE

                    n = min_num_seeds
                    
                    starv_finished=False
                    cong_finished=False

                    n_starv = n_max
                    n_cong = n_max 

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

                        trip = simul.metrics.get_aggregate_value("trips")
                        starvation = simul.metrics.get_aggregate_value("starvation")
                        congestion = simul.metrics.get_aggregate_value("congestion")
                        violation = starvation+congestion
                        
                        scale = 100 / trip
                        starvations.append(scale*starvation)
                        congestions.append(scale*congestion)
                        violations.append(scale*violation)

                        mean_trips = np.mean(trips)
                        mean_starv = np.mean(starvations)
                        mean_cong = np.mean(congestions)
                        std_starv = np.std(starvations)
                        std_cong = np.std(congestions)


                        if starv_finished==False:
                            if ci_half_length(n,alpha,std_starv) <= beta:
                                starv_finished = True
                                n_starv = n 
                        if cong_finished == False:
                            if ci_half_length(n,alpha,std_cong) <= beta:
                                cong_finished = True
                                n_cong = n
                        
                        if (starv_finished and cong_finished) or (n > n_max):
                            finished = True

                directory = 'analyses/steffen/num_sim_replications/'
                
                lock_handle = lock(directory+"output_num_reps.csv")
                
                f = open(directory+"output_num_reps.csv", "a")
                
                f.write(str(analysis_type) + ";")
                f.write(str(alpha) + ";")
                f.write(str(beta) + ";")
                f.write(str(gamma) + ";")
                f.write(str(experimental_setup["instance"]) + ";")
                f.write(str(experimental_setup["analysis"]["name"]) + ";")

                if "target_state" in experimental_setup["analysis"]:
                    f.write(str(experimental_setup["analysis"]["target_state"]))
                f.write(";")
                if "policy" in experimental_setup["analysis"]:
                    f.write(str(experimental_setup["analysis"]["policy"]))
                f.write(";")
                f.write(str(experimental_setup["analysis"]["numvehicles"]) + ";")
                f.write(str(np.mean(starvations)) + ";")
                f.write(str(np.mean(congestions)) + ";")
                f.write(str(np.mean(violations)) + ";")
                f.write(str(np.mean(trip)) + ";")
                f.write(str(np.std(starvations)) + ";")
                f.write(str(np.std(congestions)) + ";")
                f.write(str(n_starv) + ";")
                f.write(str(n_cong) + ";")
                f.write(str(round_up_to_multiple_of_n(max(n_cong,n_starv),5)) + ";")
                f.write(str(time_start)+ ";")
                f.write(str(datetime.now()-time_start)) #duration

                f.write("\n")
                f.close()

                unlock(lock_handle)



            
            

            






