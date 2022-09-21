#!/bin/python3
"""
FOMO simulator main program
"""
import copy
import sys

import settings
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike

import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen

from progress.bar import Bar

import output
import target_state
import matplotlib.pyplot as plt

import json

from helpers import *

###############################################################################

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <experimental_setup.json>")

    else:
        with open(sys.argv[1], "r") as infile:
            experimental_setup = json.load(infile)

            initial_state = init_state.get_initial_state(source=init_state.cityBike, url=experimental_setup["instance"]["url"], week=experimental_setup["instance"]["week"],
                                                         random_seed=0, number_of_stations=experimental_setup["instance"]["numstations"],
                                                         number_of_bikes=experimental_setup["instance"]["numbikes"],
                                                         target_state=getattr(target_state, experimental_setup["analysis"]["target_state"]))

            initial_state.set_seed(experimental_setup["seed"])
            initial_state.set_num_vehicles(experimental_setup["analysis"]["numvehicles"])

            if experimental_setup["analysis"]["policyargs"] != "":
                args = experimental_setup["analysis"]["policyargs"].split(",")
            else:
                args = []

            simul = sim.Simulator(
                initial_state = initial_state,
                policy = getattr(policies, experimental_setup["analysis"]["policy"])(*args),
                start_time = timeInMinutes(days=experimental_setup["instance"]["day"], hours=experimental_setup["instance"]["hour"]),
                duration = experimental_setup["duration"],
                verbose = True,
            )

            simul.run()

            lock_fd = lock("output.csv")

            f = open("output.csv", "a")

            f.write(str(experimental_setup["run"]) + ";")
            f.write(str(simul.metrics.get_aggregate_value("trips")) + ";")
            f.write(str(simul.metrics.get_aggregate_value("starvation")) + ";")
            f.write(str(simul.metrics.get_aggregate_value("congestion")))

            f.write("\n")
            f.close()

            unlock(lock_fd, stateFilename)
