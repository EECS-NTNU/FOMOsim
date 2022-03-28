#!/bin/python3

"""
Multiprocessing extension of train_value_function.py
"""
import os
import sys
from multiprocessing import Pool
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../../..'))

import policies.haflan_haga_spetalen.value_functions
import sim
from policies.haflan_haga_spetalen.analysis.train_value_function import train_value_function
import policies.haflan_haga_spetalen.settings as annsettings
import clustering.scripts


def training(input_arguments, suffix):
    SAMPLE_SIZE = 2500
    action_interval, number_of_neighbours = input_arguments

    value_function = policies.haflan_haga_spetalen.value_functions.ANNValueFunction(
        0.0001,
        annsettings.WEIGHT_INITIALIZATION_VALUE,
        annsettings.DISCOUNT_RATE,
        annsettings.VEHICLE_INVENTORY_STEP_SIZE,
        annsettings.LOCATION_REPETITION,
        annsettings.TRACE_DECAY,
        [1000, 2000, 100],
    )

    policy = policies.haflan_haga_spetalen.EpsilonGreedyValueFunctionPolicy(
        action_interval,
        number_of_neighbours,
        annsettings.EPSILON,
        value_function,
    )

    world_to_analyse = sim.Simulator(
        960,
        policy,
        policies.haflan_haga_spetalen.epsilon_greedy_value_function_policy.get_initial_state(
            entur_data_dir = "test_data",
            entur_main_file = "0900-entur-snapshot.csv",
            bike_class = "Scooter",
            number_of_scooters = SAMPLE_SIZE,
            number_of_clusters = 50,
            number_of_vans = 2,
            save = True,
            cache = True,
        ),
        verbose=False,
        visualize=False,
    )

    policy.value_function.setup(world_to_analyse.state)

    world_to_analyse.MODELS_TO_BE_SAVED=1
    world_to_analyse.TRAINING_SHIFTS_BEFORE_SAVE=50
    world_to_analyse.REPLAY_BUFFER_SIZE=100

    for cluster in world_to_analyse.state.stations:
        cluster.scooters = cluster.scooters[: round(len(cluster.scooters) * 0.6)]
    decision_times = [train_value_function(world_to_analyse, save_suffix=f"{suffix}")]

    df = pd.DataFrame(
        decision_times,
        columns=["Avg. time per shift"],
    )

    if not os.path.exists("computational_study"):
        os.makedirs("computational_study")

    df.to_excel(
        f"computational_study/training_time_ai{action_interval}_nn{number_of_neighbours}.xlsx"
    )


def multiprocess_train(function, inputs):
    with Pool() as p:
        p.starmap(function, inputs)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import itertools

    multiprocess_train(
        training,
        [
            (value, f"ai_{value[0]}_nn{value[1]}")
            for value in list(itertools.product([4], [3, 4]))
        ],
    )
