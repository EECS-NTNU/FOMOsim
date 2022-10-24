#!/bin/python3
"""
FOMO simulator, runs one experimental setup json-file
"""
import copy
import sys
from threading import Timer
from datetime import datetime

import sim
import init_state
import init_state.cityBike

import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen

import target_state

import json

from helpers import *

from multiprocessing.pool import Pool, current_process
###############################################################################

INSTANCE_DIRECTORY="instances"
LOCAL_MACHINE_TEST = False

def simulation_main(seed,state_copy,experimental_setup):
    
    print("Running seed", seed)

    state_copy.set_seed(seed) 

    sys.stdout.flush()

    simul = sim.Simulator(
        initial_state = state_copy,
        start_time = timeInMinutes(days=experimental_setup["analysis"]["day"], 
        hours=experimental_setup["analysis"]["hour"]),
        duration = experimental_setup["duration"],
        cluster = True,
        verbose = False,
    )

    simul.run()
    print("Finished running seed ", seed, ' using process ', print(current_process().name))
    
    sys.stdout.flush()

    return simul

if __name__ == "__main__":

    if LOCAL_MACHINE_TEST:
        tasks = ['experimental_setups/setup_0003.json']
    else:
        tasks = sys.argv[1:]

    for filename in tasks:
    #infile= open(filename, "r")
        with open(filename, "r") as infile: 
            print("Running file", filename) 

            time_start = datetime.now() 

            experimental_setup = json.load(infile)

            tstate = None
            if "target_state" in experimental_setup["analysis"]:
                tstate = getattr(target_state, experimental_setup["analysis"]["target_state"])

            initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + experimental_setup["instance"], 
                                                            target_state=tstate, load_from_cache=False ) #load from cache sometimes gives errors on cluster

            if experimental_setup["analysis"]["numvehicles"] > 0:
                policyargs = experimental_setup["analysis"]["policyargs"]
                policy = getattr(policies, experimental_setup["analysis"]["policy"])(**policyargs)
                initial_state.set_vehicles([policy]*experimental_setup["analysis"]["numvehicles"])

            simulations = []

            numprocesses = int(np.floor(3/4*os.cpu_count()))
            with Pool(processes=numprocesses) as pool:  #use cpu_count()
                print('Number of CPUs used:' + str(numprocesses))
                sys.stdout.flush()
                arguments = [(seed,copy.deepcopy(initial_state),experimental_setup) for seed in experimental_setup["seeds"]]
                for simul in pool.starmap(simulation_main, arguments):  #starmap_async
                    simulations.append(simul)
                    

            metric = sim.Metric.merge_metrics([sim.metrics for sim in simulations])

            lock_handle = lock("output.csv")

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
