#!/bin/python3

import copy
import time
from progress.bar import IncrementalBar
import numpy as np

import sim
import clustering.scripts
import policies
from visualization.visualizer import visualize_analysis

SEED = 2
PERIOD = 960 # 16 hours
NUMBER_OF_SCOOTERS = 100
NUMBER_OF_CLUSTERS = 10
NUMBER_OF_VANS = 1

# Set up initial state
# This is done with a script that reads data from an "entur" snapshot
state = clustering.scripts.get_initial_state(
    classname = "Scooter",
    number_of_scooters = NUMBER_OF_SCOOTERS,
    number_of_clusters = NUMBER_OF_CLUSTERS,
    number_of_vans = NUMBER_OF_VANS,
    random_seed=SEED,
)

# Set up first simulator
simulator = sim.Simulator(
    PERIOD,
    policies.DoNothing(),
    state,
    verbose=False,
    visualize=False,
    label="DoNothing",
)

# Run first simulator
simulator.run()

# # Set up initial state for simulator 2
# state2 = clustering.scripts.get_initial_state(
#     classname = "Scooter",
#     number_of_scooters = NUMBER_OF_SCOOTERS,
#     number_of_clusters = NUMBER_OF_CLUSTERS,
#     number_of_vans = NUMBER_OF_VANS,
#     random_seed=SEED,
# )

# # Set up second simulator
# simulator2 = sim.Simulator(
#     PERIOD,
#     policies.RandomActionPolicy(),
#     state2,
#     verbose=False,
#     visualize=False,
#     label="RandomAction",
# )

# simulator2.run()

# # Set up initial state for simulator 3
# state3 = clustering.scripts.get_initial_state(
#     classname = "Scooter",
#     number_of_scooters = NUMBER_OF_SCOOTERS,
#     number_of_clusters = NUMBER_OF_CLUSTERS,
#     number_of_vans = NUMBER_OF_VANS,
#     random_seed=SEED,
# )

# # Set up second simulator
# simulator3 = sim.Simulator(
#     PERIOD,
#     policies.RebalancingPolicy(),
#     state3,
#     verbose=False,
#     visualize=False,
#     label="Rebalancing",
# )

# simulator3.run()

# Visualize results
visualize_analysis([simulator])
