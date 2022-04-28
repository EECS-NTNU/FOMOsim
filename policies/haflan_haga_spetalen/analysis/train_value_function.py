#!/bin/python3

"""
MAIN SCRIPT FOR TRAINING THE MODEL
"""

import copy
import time
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../../..'))

import sim
import clustering.scripts
import policies
import policies.haflan_haga_spetalen
import policies.haflan_haga_spetalen.epsilon_greedy_value_function_policy
import policies.haflan_haga_spetalen.settings as annsettings
import policies.haflan_haga_spetalen.value_functions
import ideal_state

import policies.haflan_haga_spetalen.training_simulation.scripts
from progress.bar import IncrementalBar

def get_train_directory(simulator, suffix=None):
    suffix = suffix if suffix else f"{simulator.created_at}"
    return (
        f"trained_models/{simulator.policy.value_function.__repr__()}/"
        f"c{len(simulator.state.stations)}_s{len(simulator.state.get_scooters())}/{suffix}"
    )

def train_value_function(
    world, save_suffix="", scenario_training=True, epsilon_decay=True
):
    """
    Main method for training any value function attached to world object
    :param world: world object with l
    :param save_suffix:
    :param scenario_training:
    :param epsilon_decay:
    :return:
    """
    # Add progress bar
    progress_bar = IncrementalBar(
        "Training value function",
        check_tty=False,
        max=(world.TRAINING_SHIFTS_BEFORE_SAVE * world.MODELS_TO_BE_SAVED),
        suffix="%(percent)d%% - ETA %(eta)ds",
    )
    print(
        f"-------------------- {world.policy.value_function.__str__()} training --------------------"
    )
    # Determine number of shifts to run based on the training parameters
    number_of_shifts = world.TRAINING_SHIFTS_BEFORE_SAVE * world.MODELS_TO_BE_SAVED
    # Initialize the epsilon parameter
    world.policy.epsilon = annsettings.INITIAL_EPSILON if epsilon_decay else world.EPSILON
    training_times = []
    for shift in range(number_of_shifts + 1):
        # Start timer for computational time analysis
        start = time.time()
        policy_world = copy.deepcopy(world)
        # Update shifts trained counter
        policy_world.policy.value_function.update_shifts_trained(shift)
        if epsilon_decay and shift > 0:
            # Decay epsilon
            policy_world.policy.epsilon -= (
                annsettings.INITIAL_EPSILON - annsettings.FINAL_EPSILON
            ) / number_of_shifts
        if shift % world.TRAINING_SHIFTS_BEFORE_SAVE == 0:
            # Save model for intervals set by training parameters
            policy_world.save_sim(
                cache_directory=get_train_directory(policy_world, save_suffix), suffix=shift
            )

        # avoid running the world after the last model is saved
        if shift != number_of_shifts:
            if scenario_training:
                # scenario training is a faster simulation engine used when learning
                policies.haflan_haga_spetalen.training_simulation.scripts.training_simulation(policy_world)
            else:
                # Train using event based simulation engine
                policy_world.run()
            # Save the policy for "master" world copy
            world.policy = policy_world.policy
            progress_bar.next()
        # stop timer
        training_times.append(time.time() - start)

    return sum(training_times) / number_of_shifts

def get_time(day=0, hour=0, minute=0):
    return 24*60*day + 60*hour + minute

if __name__ == "__main__":
    import pandas as pd
    import os

    SAMPLE_SIZE = 2500
    NUMBER_OF_CLUSTERS = [10, 20, 30, 50]
    decision_times = []
    for num_clusters in NUMBER_OF_CLUSTERS:
        value_function = policies.haflan_haga_spetalen.value_functions.ANNValueFunction(
            0.0001,
            annsettings.WEIGHT_INITIALIZATION_VALUE,
            annsettings.DISCOUNT_RATE,
            annsettings.VEHICLE_INVENTORY_STEP_SIZE,
            annsettings.LOCATION_REPETITION,
            annsettings.TRACE_DECAY,
            [1000, 2000, 1000, 200],
        )

        policy = policies.haflan_haga_spetalen.EpsilonGreedyValueFunctionPolicy(
            annsettings.DIVIDE_GET_POSSIBLE_ACTIONS,
            annsettings.NUMBER_OF_NEIGHBOURS,
            annsettings.EPSILON,
            value_function,
        )

        state = clustering.scripts.get_initial_state("test_data", "0900-entur-snapshot.csv", "Scooter", number_of_scooters = 250, number_of_clusters = 5, number_of_vans = 1, random_seed = 1)
        state.simulation_scenarios = policies.haflan_haga_spetalen.generate_scenarios(state)
        istate = ideal_state.evenly_distributed_ideal_state(state)
        state.set_ideal_state(istate)

        world_to_analyse = sim.Simulator(
            960,
            policy,
            state,
            start_time = get_time(day=1, hour=7),
            verbose=False,
        )

        policy.value_function.setup(world_to_analyse.state)

        world_to_analyse.MODELS_TO_BE_SAVED=1
        world_to_analyse.TRAINING_SHIFTS_BEFORE_SAVE=10
        world_to_analyse.REPLAY_BUFFER_SIZE=64
        
        decision_times.append(train_value_function(world_to_analyse))

    df = pd.DataFrame(
        decision_times,
        index=NUMBER_OF_CLUSTERS,
        columns=["Avg. time per shift"],
    )

    if not os.path.exists("computational_study"):
        os.makedirs("computational_study")

    df.to_excel("computational_study/training_time_clusters_shift_short.xlsx")
