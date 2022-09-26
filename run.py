#!/bin/python3
"""
FOMO simulator, runs one experimental setup json-file
"""
import copy
import sys

import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike

import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen

import target_state

import json

from helpers import *

###############################################################################

if __name__ == "__main__":

    for filename in sys.argv[1:]:
        with open(filename, "r") as infile:
            print("Running file", filename) #filename='experimental_setups/setup_0063.json'

            experimental_setup = json.load(infile)

            initial_state = init_state.get_initial_state(source=init_state.cityBike, url=experimental_setup["instance"]["url"], week=experimental_setup["instance"]["week"],
                                                         random_seed=0, number_of_stations=experimental_setup["instance"]["numstations"],
                                                         number_of_bikes=experimental_setup["instance"]["numbikes"],
                                                         target_state=getattr(target_state, experimental_setup["analysis"]["target_state"]))

            simulations = []

            for seed in experimental_setup["seeds"]:
                print("Running seed", seed)

                state_copy = copy.deepcopy(initial_state)
                state_copy.set_seed(seed)
                state_copy.set_num_vehicles(experimental_setup["analysis"]["numvehicles"])

                if experimental_setup["analysis"]["policyargs"] != "":
                    args = experimental_setup["analysis"]["policyargs"].split(",")
                else:
                    args = []
                
                exp_policy = getattr(policies, experimental_setup["analysis"]["policy"])(*args)
                if experimental_setup["analysis"]["crit_weights"] != "":
                    [w1,w2,w3,w4] = experimental_setup["analysis"]["crit_weights"]
                    exp_policy.set_criticality_weights(w1,w2,w3,w4)
                simul = sim.Simulator(
                    initial_state = state_copy,
                    policy =exp_policy,
                    start_time = timeInMinutes(days=experimental_setup["instance"]["day"], hours=experimental_setup["instance"]["hour"]),
                    duration = experimental_setup["duration"],
                    verbose = False,
                )

                simul.run()

                simulations.append(simul)

            metric = sim.Metric.merge_metrics([sim.metrics for sim in simulations])

            lock_handle = lock("output.csv")

            f = open("output.csv", "a")

            f.write(str(experimental_setup["run"]) + ";")
            f.write(str(experimental_setup["instance"]["name"]) + ";")
            f.write(str(experimental_setup["analysis"]["name"]) + ";")
            f.write(str(metric.get_aggregate_value("trips")) + ";")
            f.write(str(metric.get_aggregate_value("starvation")) + ";")
            f.write(str(metric.get_aggregate_value("congestion")) + ";")
            f.write(str(metric.get_aggregate_value("starvation_stdev")) + ";")
            f.write(str(metric.get_aggregate_value("congestion_stdev")))

            f.write("\n")
            f.close()

            unlock(lock_handle)

    print("Done")
    sys.stdout.flush()
