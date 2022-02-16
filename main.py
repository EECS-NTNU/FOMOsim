#!/bin/python3

import copy
import time
from progress.bar import IncrementalBar
import numpy as np

import sim
import clustering.scripts
import policies
from visualization.visualizer import visualize_analysis

# Set up initial state
# This is done with a script that reads data from an "entur" snapshot
state = clustering.scripts.get_initial_state(
    classname = "Bike",
    sample_size = 50,
    number_of_clusters = 5,
    initial_in_use = 10,
    number_of_vans = 1,
)

# Set up first simulator
simulator = sim.Simulator(
    100,
    policies.RandomActionPolicy(),
    copy.deepcopy(state),
    verbose=False,
    visualize=False,
    label="Rebalancing",
)

# Run first simulator
simulator.run()

# # Set up second simulator with different policy
# simulator2 = sim.Simulator(
#     40,
#     policies.DoNothing(),
#     copy.deepcopy(state),
#     verbose=True,
#     visualize=False,
#     label="DoNothing",
# )

# # Run second simulator
# simulator2.run()

# Visualize results
visualize_analysis([simulator])
