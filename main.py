#!/bin/python3

import copy
import time
from progress.bar import IncrementalBar
import numpy as np

import sim
import clustering.scripts
import policies
import policies.epsilon_greedy_value_function_policy
import policies.epsilon_greedy_value_function_policy.epsilon_greedy_value_function_policy
import policies.epsilon_greedy_value_function_policy.settings as annsettings
import policies.epsilon_greedy_value_function_policy.value_functions
from visualization.visualizer import visualize_analysis

# Set up initial state
# This is done with a script that reads data from an "entur" snapshot
state = policies.epsilon_greedy_value_function_policy.epsilon_greedy_value_function_policy.get_initial_state(
    500,
    5,
    number_of_vans=1,
)

# Set up first simulator

# value_function = decision.value_functions.ANNValueFunction(
#         annsettings.ANN_LEARNING_RATE,
#         annsettings.WEIGHT_INITIALIZATION_VALUE,
#         annsettings.DISCOUNT_RATE,
#         annsettings.VEHICLE_INVENTORY_STEP_SIZE,
#         annsettings.LOCATION_REPETITION,
#         annsettings.TRACE_DECAY,
#         [100, 100, 100],
# )

value_function = policies.epsilon_greedy_value_function_policy.value_functions.LinearValueFunction(
        annsettings.WEIGHT_UPDATE_STEP_SIZE,
        annsettings.WEIGHT_INITIALIZATION_VALUE,
        annsettings.DISCOUNT_RATE,
        annsettings.VEHICLE_INVENTORY_STEP_SIZE,
        annsettings.LOCATION_REPETITION,
        annsettings.TRACE_DECAY,
)
policy = policies.epsilon_greedy_value_function_policy.EpsilonGreedyValueFunctionPolicy(
    annsettings.DIVIDE_GET_POSSIBLE_ACTIONS,
    annsettings.NUMBER_OF_NEIGHBOURS,
    annsettings.EPSILON,
    value_function,
)

simulator = sim.Simulator(
    10080,
    policy,
    copy.deepcopy(state),
    verbose=True,
    visualize=False,
    label="EGVFP",
)

policy.value_function.setup(simulator.state)

simulator.run()

# Visualize results
visualize_analysis([simulator])
