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
import demand

import target_state

import json

from helpers import *

from multiprocessing.pool import Pool
from multiprocessing import current_process

from create_runs_base_settings import *


###############################################################################

INSTANCE_DIRECTORY="instances"
LOCAL_MACHINE_TEST = False

CPU_FACTOR = 2/10   # I got some memory issues at 3/4. Maybe think about why we get these issues? 
MAX_CPU = 8

def simulation_main(seed,state_copy,experimental_setup):
    
    print("Running seed", seed)

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
        cluster = True,
        verbose = False,
    )

    simul.run()
    print("Finished running seed ", seed, ' using process ', print(current_process().name))
    
    scale = 100 / simul.metrics.get_aggregate_value("trips")
    perc_lost_trips = scale*(simul.metrics.get_aggregate_value("starvation")+
                        simul.metrics.get_aggregate_value("congestion"))
    simul.metrics.add_metric(simul,'perc_lost_trips',perc_lost_trips)

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

            initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + experimental_setup["instance"])
                                                            #number_of_stations,number_of_bikes

            if experimental_setup["analysis"]["numvehicles"] > 0:
                policyargs = experimental_setup["analysis"]["policyargs"]
                policy = getattr(policies, experimental_setup["analysis"]["policy"])(**policyargs)
                initial_state.set_vehicles([policy]*experimental_setup["analysis"]["numvehicles"])

            simulations = []

            #---------------WITH parallelization------------------

            # numprocesses = min(MAX_CPU,int(np.floor(CPU_FACTOR*os.cpu_count())))
            # with Pool(processes=numprocesses) as pool:  #use cpu_count()
            #     print('Number of CPUs used:' + str(numprocesses))
            #     sys.stdout.flush()
            #     arguments = [(seed,copy.deepcopy(initial_state),experimental_setup) for seed in experimental_setup["seeds"]]
            #     for simul in pool.starmap(simulation_main, arguments):  #starmap_async
            #         simulations.append(simul)

            #---------------WITHOUT parallelization------------------


            for seed in experimental_setup["seeds"]:
                simul = simulation_main(copy.deepcopy(seed,copy.deepcopy(initial_state),experimental_setup))
                simulations.append(simul)   

            #----------------------------------------------

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
            f.write(str(round(metric.get_aggregate_value("starvation"),2)) + ";")
            f.write(str(round(metric.get_aggregate_value("congestion"),2)) + ";")
            f.write(str(round(metric.get_aggregate_value("starvation_stdev"),2)) + ";")
            f.write(str(round(metric.get_aggregate_value("congestion_stdev"),2))+ ";")
            f.write(str(time_start)+ ";")
            f.write(str(datetime.now()-time_start)) #duration

            f.write("\n")
            f.close()

            unlock(lock_handle)

    print("Done")
    sys.stdout.flush()
