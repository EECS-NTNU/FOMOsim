#!/bin/python3
"""
FOMO simulator, runs one experimental setup json-file
"""
import copy
import sys
from threading import Timer
from datetime import datetime
import os

import sim
import init_state
import init_state.cityBike

import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import demand

import target_state

import json

from helpers import *

from multiprocessing.pool import Pool
from multiprocessing import Manager

import numpy as np



###############################################################################

INSTANCE_DIRECTORY="instances"
EXPERIMENTS_DIRECTORY = "experimental_setups"
filenames = [EXPERIMENTS_DIRECTORY+'/'+exp for exp in os.listdir("experimental_setups")]

#######

def main_code(windows_lock,filename):

    with open(filename, "r") as infile: #infile= open(filename, "r")
        print("Running file", filename) #filename='experimental_setups/setup_0003.json'

        time_start = datetime.now() 

        experimental_setup = json.load(infile)

        initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + experimental_setup["instance"])
                                                            #number_of_stations,number_of_bikes

        if experimental_setup["analysis"]["numvehicles"] > 0:
            policyargs = experimental_setup["analysis"]["policyargs"]
            policy = getattr(policies, experimental_setup["analysis"]["policy"])(**policyargs)
            initial_state.set_vehicles([policy]*experimental_setup["analysis"]["numvehicles"])

        simulations = []
        for seed in experimental_setup["seeds"]:
            #print("Running seed", seed)

            state_copy = copy.deepcopy(initial_state)
            state_copy.set_seed(seed)

            sys.stdout.flush()

            trgt_state = None
            if experimental_setup["analysis"]["numvehicles"] > 0:
                trgt_state = getattr(target_state, experimental_setup["analysis"]["target_state"])()

            simul = sim.Simulator(
                initial_state = state_copy,
                target_state = trgt_state,
                demand = demand.Demand(),
                start_time = timeInMinutes( days=experimental_setup["analysis"]["day"], 
                                            hours=experimental_setup["analysis"]["hour"]),
                duration = experimental_setup["duration"],
                cluster = False,
                verbose = False,
            )

            simul.run()

            simulations.append(simul)

        metric = sim.Metric.merge_metrics([sim.metrics for sim in simulations])

        lock_handle = lock("output.csv")

        with windows_lock:
            f = open("output.csv", "a")

            f.write(str(experimental_setup["run"]) + ";")
            f.write(str(experimental_setup["instance"]) + ";")
            f.write(str(experimental_setup["analysis"]["name"]) + ";")
            if "target_state" in experimental_setup["analysis"]:
                f.write(str(experimental_setup["analysis"]["target_state"]))
            f.write(";")
            if "policy" in experimental_setup["analysis"]:
                f.write(str(experimental_setup["analysis"]["policy"]))
            f.write(";")
            f.write(str(experimental_setup["analysis"]["numvehicles"]) + ";")
            f.write(str(metric.get_aggregate_value("trips")) + ";")
            f.write(str(metric.get_aggregate_value("starvation")) + ";")
            f.write(str(metric.get_aggregate_value("congestion")) + ";")
            f.write(str(metric.get_aggregate_value("starvation_stdev")) + ";")
            f.write(str(metric.get_aggregate_value("congestion_stdev"))+ ";")
            f.write(str(time_start)+ ";")
            f.write(str(datetime.now()-time_start)) #duration

            f.write("\n")
            f.close()

        unlock(lock_handle)

        print("Done")
        sys.stdout.flush()

        string_result = 'finished with instance '+str(experimental_setup["instance"])+' and analysis '+str(experimental_setup["analysis"]["name"])
        print(string_result)
        return string_result

if __name__ == "__main__":
    with Manager() as manager:
        lock = manager.Lock()
        # https://superfastpython.com/multiprocessing-pool-mutex-lock/
        # https://superfastpython.com/multiprocessing-mutex-lock-in-python/
        numprocesses = int(np.floor(3/4*os.cpu_count()))
        with Pool(processes=numprocesses) as pool:  #use cpu_count()
            print('Number of CPUs used:' + str(numprocesses))
            arguments = [(lock,filename) for filename in filenames]
            for finished_string in pool.starmap(main_code, arguments):  #starmap_async
                print(f'Got result: {finished_string}', flush=True)
    
    
