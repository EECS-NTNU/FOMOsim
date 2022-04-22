#!/bin/python3

import copy
import time
# from progress.bar import IncrementalBar
import numpy as np

import settings
import sim
import clustering.scripts
import tripStats.parse

# import GUI.dashboard

# from tripStats.parse import calcDistances, get_initial_state

import policies
import policies.fosen_haldorsen
import policies.gleditsch_hagen
from visualization.visualizer import visualize_analysis
import ideal_state

# from tripStats.helpers import printTime

from GUI.dashboard import GUI_main

simulators = []

def get_time(day=0, hour=0, minute=0):
    return 24*60*day + 60*hour + minute

WEEK = 30
START_DAY = 2
START_HOUR = 8
PERIOD = get_time(0, 16)

###############################################################################

# Set up initial state

# calcDistances(city = "Oslo") # To ensure that station.txt is available
# TODO skriv som assume i interface

if settings.USER_INTERFACE_MODE == "CMD" or not GUI_main():

    #state, _ = tripStats.parse.get_initial_state(city="Oslo", week=WEEK)
    state = clustering.scripts.get_initial_state("test_data", "0900-entur-snapshot.csv", "Bike", number_of_scooters = 250, number_of_clusters = 5, number_of_vans = 2, random_seed = 1)

    ###############################################################################
    # calculate ideal state

    #ideal_state = ideal_state.evenly_distributed_ideal_state(state)
    ideal_state = ideal_state.outflow_ideal_state(state)
    state.set_ideal_state(ideal_state)

    ###############################################################################

    # Set up simulator
    simulators.append(sim.Simulator(
        PERIOD,
        policies.gleditsch_hagen.GleditschHagenPolicy(),
        copy.deepcopy(state),
        verbose=True,
        start_time = get_time(day=START_DAY, hour=START_HOUR),
        label="GleditschHagen",
    ))

    # Run first simulator
    simulators[-1].run()

    ###############################################################################

    # Visualize results
    visualize_analysis(simulators)
