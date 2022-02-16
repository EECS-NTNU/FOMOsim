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
    classname = "Scooter",
    number_of_scooters = 500,
    number_of_clusters = 10,
    number_of_vans = 1,
)

# Set up first simulator
simulator = sim.Simulator(
    1440,
    policies.DoNothing(),
    state,
    verbose=False,
    visualize=False,
    label="DoNothing",
)

# Run first simulator
simulator.run()

# Set up initial state
# This is done with a script that reads data from an "entur" snapshot
state2 = clustering.scripts.get_initial_state(
    classname = "Scooter",
    number_of_scooters = 500,
    number_of_clusters = 10,
    number_of_vans = 1,
)

# Set up first simulator
simulator2 = sim.Simulator(
    1440,
    policies.RandomActionPolicy(),
    state2,
    verbose=False,
    visualize=False,
    label="RandomAction",
)

simulator2.run()

# Visualize results
visualize_analysis([simulator, simulator2])
