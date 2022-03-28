#!/bin/python3

import copy
import time
from progress.bar import IncrementalBar
import numpy as np

import sim
import clustering.scripts

from tripStats.parse import calcDistances, get_initial_state

import policies
import policies.fosen_haldorsen
from visualization.visualizer import visualize_analysis
import ideal_state

from tripStats.helpers import printTime

PERIOD = 960 # 16 hours

simulators = []

###############################################################################

# Set up initial state

# calcDistances(city = "Oslo") # To ensure that station.txt is available
printTime()
print(" * get_initial_state()")

state = get_initial_state(city = "Oslo", week=12)

# state = clustering.scripts.get_initial_state(
#     "test_data",
#     "0900-entur-snapshot.csv",
#     "Scooter",
#     number_of_scooters = 2500,
#     number_of_clusters = 10,
#     number_of_vans = 2,
#     random_seed = 1,
# )

###############################################################################
# calculate ideal state

ideal_state = ideal_state.evenly_distributed_ideal_state(state)
state.set_ideal_state(ideal_state)
#ideal_state.haflan_haga_spetalen_ideal_state(state)

###############################################################################

# # Set up simulator
# simulators.append(sim.Simulator(
#     PERIOD,
#     policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False),
#     copy.deepcopy(state),
#     verbose=True,
#     label="FosenHaldorsen",
# ))

# # Run first simulator
# simulators[-1].run()

###############################################################################

# # Set up simulator
# simulators.append(sim.Simulator(
#     PERIOD,
#     policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True),
#     copy.deepcopy(state),
#     verbose=True,
#     label="Greedy",
# ))

# # Run first simulator
# simulators[-1].run()

###############################################################################

printTime()
print(" * sim do nothing")

# Set up simulator
simulators.append(sim.Simulator(
    PERIOD,
    policies.DoNothing(),
    copy.deepcopy(state),
    verbose=True,
    label="DoNothing",
))

# Run first simulator
simulators[-1].run()

###############################################################################

printTime()
print(" * sim rebalancing")

# Set up simulator
simulators.append(sim.Simulator(
    PERIOD,
    policies.RebalancingPolicy(),
    copy.deepcopy(state),
    verbose=True,
    label="Rebalancing",
))

# Run first simulator
simulators[-1].run()


###############################################################################

# Visualize results
printTime()
print(" * start visualize")
visualize_analysis(simulators)

printTime()
