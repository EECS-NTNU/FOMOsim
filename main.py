#!/bin/python3

import copy
import time
from progress.bar import IncrementalBar
import numpy as np

import sim
import clustering.scripts
import policies
from visualization.visualizer import visualize_analysis

PERIOD = 960 # 16 hours

###############################################################################

# Set up initial state
# This is done with a script that reads data from an "entur" snapshot

state = clustering.scripts.get_initial_state(
    classname = "Scooter",
    number_of_scooters = 1000,
    number_of_clusters = 10,
    number_of_vans = 1,
    random_seed=1,
)

###############################################################################

# Set up first simulator
simulator = sim.Simulator(
    PERIOD,
    policies.DoNothing(),
    copy.deepcopy(state),
    label="DoNothing",
)

# Run first simulator
simulator.run()

###############################################################################

# Set up second simulator
simulator2 = sim.Simulator(
    PERIOD,
    policies.RandomActionPolicy(),
    copy.deepcopy(state),
    label="RandomAction",
)

# Run second simulator
simulator2.run()

###############################################################################

# Set up third simulator
simulator3 = sim.Simulator(
    PERIOD,
    policies.RebalancingPolicy(),
    copy.deepcopy(state),
    label="Rebalancing",
)

# Run third simulator
simulator3.run()

###############################################################################

# Visualize results
visualize_analysis([simulator, simulator2, simulator3])
