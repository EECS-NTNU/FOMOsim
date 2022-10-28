####################################
# set the right working directory #
###################################

import os 
import sys
from pathlib import Path

desired_path = Path(__file__).parents[3]
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
import demand

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
analysis_type = 'absolute_iterate' #relative1,absolute,absolute_iterate

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

INSTANCE_DIRECTORY="instances"

#perform the following analysis for two different policies!!!
analysis = dict(name="outflow",
         target_state="outflow_target_state",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6)

if __name__ == "__main__":

    results = None

    n_starvations = []
    n_congestions = []
    n_violations = []

    mean_starv = []
    mean_cong = []
    mean_violations = []
    mean_trips = []
    
    std_starv = []
    std_cong = []

    for instance in instances:
        print("  instance: ", instance)


        tstate = None
        if "target_state" in analysis:
            tstate = getattr(target_state, analysis["target_state"])()

        initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + instance)

        if analysis["numvehicles"] > 0:
            policyargs = analysis["policyargs"]
            policy = getattr(policies, analysis["policy"])(**policyargs)
            initial_state.set_vehicles([policy]*analysis["numvehicles"])

        
        
        if analysis_type == 'relative1':
        

            #FIRST DO THE MINIMUM NUMBER OF SEEDS
            
            starvations = []
            congestions = []
            for seed in range(min_num_seeds):
                #seed = 0

                state_copy = copy.deepcopy(initial_state)
                state_copy.set_seed(seed)

                simul = sim.Simulator(
                    initial_state = state_copy,
                    target_state = tstate,
                    demand = demand.Demand(),
                    start_time = timeInMinutes(days=analysis["day"], hours=analysis["hour"]),
                    duration = DURATION,
                    verbose = True,
                )
                
                simul.run()

                scale = 100 / simul.metrics.get_aggregate_value("trips")
                starvations.append(scale*simul.metrics.get_aggregate_value("starvation"))
                congestions.append(scale*simul.metrics.get_aggregate_value("congestion"))
                

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
                    normalized_half = ci_half_length(n,alpha,std_starv) / np.abs(mean_starv)
                    if normalized_half <= gamma_star:
                        starv_finished = True
                        n_starv = n 

                if cong_finished == False:
                    normalized_half = ci_half_length(n,alpha,std_cong) / np.abs(mean_cong)
                    if normalized_half <= gamma_star:
                        cong_finished = True
                        n_cong = n
                
                if starv_finished and cong_finished:
                    finished = True

                if n > n_max:
                    finished = True

                if finished:
                    n_starvations.append(n_starv)
                    n_congestions.append(n_cong)

        elif analysis_type == 'absolute':

            starvations = []
            congestions = []
            violations = []
            trips = []

            for seed in range(n_max):
                #seed = 0

                state_copy = copy.deepcopy(initial_state)
                state_copy.set_seed(seed)

                simul = sim.Simulator(
                    initial_state = state_copy,
                    target_state=tstate,
                    demand=demand.Demand(),
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

            n_starvations.append(approximate_num_reps_absolute(starvations,beta,alpha,5))
            n_congestions.append(approximate_num_reps_absolute(congestions,beta,alpha,5))
            n_violations.append(approximate_num_reps_absolute(violations,beta,alpha,5))
            
            mean_starv.append(np.mean(starvations))
            mean_cong.append(np.mean(congestions))
            mean_violations.append(np.mean(violations))
            mean_trips.append(np.mean(trip))

            std_starv.append(np.std(starvations))
            std_cong.append(np.std(congestions))
            
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
                        n_starvations.append(n_starv)
                if cong_finished == False:
                    if ci_half_length(n,alpha,std_cong) <= beta:
                        cong_finished = True
                        n_cong = n
                        n_congestions.append(n_cong)
                
                if (starv_finished and cong_finished) or (n > n_max):
                    finished = True
            
                    mean_starv.append(np.mean(starvations))
                    mean_cong.append(np.mean(congestions))
                    mean_violations.append(np.mean(violations))
                    mean_trips.append(np.mean(trip))

                    std_starv.append(np.std(starvations))
                    std_cong.append(np.std(congestions))


    results = pd.DataFrame(list(zip(instances, n_starvations,n_congestions,n_violations,mean_starv,std_starv,mean_cong,std_cong,mean_violations,mean_trips)),
               columns =['instance', 'n_starv','n_cong','n_viol','mean_starv','std_starv','mean_cong','std_cong','mean_viol','mean_trips'])
    print(results)
    
    directory = 'analyses/steffen/num_sim_replications/'
    if analysis_type == 'absolute':
        filename = 'num_rep_analysis_beta'+str(beta)+'_alpha'+str(alpha)
    else:
        filename = 'num_rep_analysis_gamma'+str(gamma)+'_alpha'+str(alpha)
    results.to_csv(directory+filename+'.csv')

            
            

            






