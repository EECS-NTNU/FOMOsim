#!/bin/python3
"""
FOMO simulator, runs one experimental setup json-file
"""
import copy
import sys
from threading import Timer
from datetime import datetime
import json

import sim
import init_state
import init_state.cityBike
import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import target_state
import demand
from helpers import *

###############################################################################

INSTANCE_DIRECTORY="instances"

if __name__ == "__main__":

    for filename in sys.argv[1:]:
        with open(filename, "r") as infile: #infile= open(filename, "r")
            print("Running file", filename) #filename='experimental_setups/setup_0063.json'

            time_start = datetime.now() 

            experimental_setup = json.load(infile)

            # set up target state
            tstate = None
            if "target_state" in experimental_setup["analysis"]:
                tstate = getattr(target_state, experimental_setup["analysis"]["target_state"])()

            # set up initial state
            initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + experimental_setup["instance"])

            # set up vehicles
            if experimental_setup["analysis"]["numvehicles"] > 0:
                policyargs = experimental_setup["analysis"]["policyargs"]
                policy = getattr(policies, experimental_setup["analysis"]["policy"])(**policyargs)
                initial_state.set_sb_vehicles([policy]*experimental_setup["analysis"]["numvehicles"])

            # run simulations, one for each seed
            simulations = []
            for seed in experimental_setup["seeds"]:
                print("Running seed", seed)

                state_copy = copy.deepcopy(initial_state)
                state_copy.set_seed(seed)

                sys.stdout.flush()

                simul = sim.Simulator(
                    initial_state = state_copy,
                    target_state = tstate,
                    demand = demand.Demand(),
                    start_time = timeInMinutes(days=experimental_setup["analysis"]["day"], hours=experimental_setup["analysis"]["hour"]),
                    duration = experimental_setup["duration"],
                    cluster = True,
                    verbose = False,
                )

                simul.run()

                simulations.append(simul)

            # record output

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
