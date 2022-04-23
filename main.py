# !/bin/python3
"""
FOMO simulator main program
"""
import copy

import settings
import sim
import clustering.scripts
import tripStats.parse

import policies
import policies.fosen_haldorsen
from visualization.visualizer import visualize_analysis
import ideal_state

from GUI.dashboard import GUI_main
from tripStats.helpers import dateAndTimeStr

simulators = []

def get_time(day, hour):
    return 60*24*day + 60*hour

WEEK = 30
START_DAY = 2
START_HOUR = 8
PERIOD = get_time(0, 16)

###############################################################################
# Set up initial state

if settings.USER_INTERFACE_MODE == "CMD" or not GUI_main():

    settings.TRAFFIC_LOGGING = False # since it slows down simulation a lot

    state, _ = tripStats.parse.get_initial_state(city="Oslo", week=WEEK)
    #state = clustering.scripts.get_initial_state("test_data", "0900-entur-snapshot.csv", "Scooter", number_of_scooters = 250, number_of_clusters = 10, number_of_vans = 2, random_seed = 1)

    ###############################################################################
    # calculate ideal state

    #ideal_state = ideal_state.evenly_distributed_ideal_state(state)
    ideal_state = ideal_state.outflow_ideal_state(state)
    state.set_ideal_state(ideal_state)
    ###############################################################################

    print("before 3 simulations", dateAndTimeStr)

    # Set up simulator
    simulators.append(sim.Simulator(
        PERIOD,
        policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False),
        copy.deepcopy(state),
        verbose=True,
        start_time = get_time(day=START_DAY, hour=START_HOUR),
        label="Fosen&Haldorsen",
    ))

    # Run first simulator
    simulators[-1].run()

    ###############################################################################

    # Set up simulator
    simulators.append(sim.Simulator(
        PERIOD,
        policies.DoNothing(),
        copy.deepcopy(state),
        verbose=True,
        start_time = get_time(day=START_DAY, hour=START_HOUR),
        label="DoNothing",
    ))

    # Run first simulator
    simulators[-1].run()

    ###############################################################################

    # Set up simulator
    simulators.append(sim.Simulator(
        PERIOD,
        policies.RebalancingPolicy(),
        copy.deepcopy(state),
        verbose=True,
        start_time = get_time(day=START_DAY, hour=START_HOUR),
        label="Rebalancing",
    ))

    # Run first simulator
    simulators[-1].run()

    ###############################################################################

    print("after 3 simulations", dateAndTimeStr)

    # Visualize results
    visualize_analysis(simulators)
