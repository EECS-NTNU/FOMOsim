#!/bin/python3

import copy
import time
from progress.bar import IncrementalBar
import numpy as np

import sim
import clustering.scripts
import policies
from visualization.visualizer import visualize_analysis

###############################################################################

# Set up initial state
# This is done with a script that reads data from an "entur" snapshot

SEED = 2
PERIOD = 960 # 16 hours
NUMBER_OF_SCOOTERS = 1000
NUMBER_OF_CLUSTERS = 10
NUMBER_OF_VANS = 1

state = clustering.scripts.get_initial_state(
    classname = "Scooter",
    number_of_scooters = NUMBER_OF_SCOOTERS,
    number_of_clusters = NUMBER_OF_CLUSTERS,
    number_of_vans = NUMBER_OF_VANS,
    random_seed=SEED,
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
