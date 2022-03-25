#!/bin/python3

import copy
import time
from progress.bar import IncrementalBar
import numpy as np

import sim
import clustering.scripts

from tripStats.parse import get_initial_state 

import policies
from visualization.visualizer import visualize_analysis

PERIOD = 960 # 16 hours

simulators = []

###############################################################################

# Set up initial state

# # This is done with a script that reads data from an "entur" snapshot
# state = clustering.scripts.get_initial_state(
#     entur_data_dir = "test_data",
#     entur_main_file = "0900-entur-snapshot.csv",
#     bike_class = "Scooter",
#     number_of_scooters = 1000,
#     number_of_clusters = 10,
#     number_of_vans = 2,
#     random_seed=1,
# )

# state = tripStats.get_initial_state("Oslo", 20)

# *****************WORKS FINE in branch tripStats 
# This is set up manually

#state = tripStats.get_initial_state("Oslo", 20) # format wanted by Asbjørn
state = get_initial_state("Oslo", 20) # parse.py will crash in line 250(?) "    return sim.State.get_initial_state("

state = sim.State.get_initial_state(
    bike_class = "Bike", # was "Scooter"
    distance_matrix = [
        [0, 4, 2, 3],
        [4, 0, 5, 1],
        [2, 5, 0, 4],
        [3, 1, 4, 0],
    ],
    main_depot = None,
    secondary_depots = [],
#    number_of_scooters = [0, 0, 2, 4],
    number_of_scooters = [0, 0, 0, 1],
    arrive_intensities = [0, 0, 2, 5],
#    leave_intensities = [0, 0, 5, 2],
    leave_intensities = [5, 5, 5, 5],
    move_probabilities = np.zeros((4, 4), dtype="float64"),
#    number_of_vans = 2,
    number_of_vans = 0,
    random_seed = 1,
)
    
###############################################################################

# Set up first simulator
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

# Set up second simulator
simulators.append(sim.Simulator(
    PERIOD,
    policies.RandomActionPolicy(),
    copy.deepcopy(state),
    verbose=True,
    label="RandomAction",
))

# Run second simulator
simulators[-1].run()



###############################################################################

# # The rebalancing policy needs special information in the state object ("ideal state" calculations)
# # We can use the following function to generate this kind of state object
# state_with_ideal_state = policies.epsilon_greedy_value_function_policy.epsilon_greedy_value_function_policy.get_initial_state(
#     entur_data_dir = "test_data",
#     entur_main_file = "0900-entur-snapshot.csv",
#     bike_class = "Scooter",
#     number_of_scooters = 1000,
#     number_of_clusters = 10,
#     number_of_vans = 2,
#     random_seed=1,
# )

# # Set up third simulator
# simulators.append(sim.Simulator(
#     PERIOD,
#     policies.RebalancingPolicy(),
#     copy.deepcopy(state_with_ideal_state),
#     verbose=True,
#     label="Rebalancing",
# ))

# # Run third simulator
# simulators[-1].run()

###############################################################################

# Visualize results

visualize_analysis(simulators)
