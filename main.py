#!/bin/python3
"""
FOMO simulator main program
"""

import settings
import init_state
import ideal_state
import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
#import policies.gleditsch_hagen
import sim
import visualization
from GUI.dashboard import GUI_main

def get_time(day=0, hour=0, minute=0):
    return 24*60*day + 60*hour + minute

if settings.USER_INTERFACE_MODE == "CMD" or not GUI_main():

    WEEK = 30

    ###############################################################################
    # get initial state

    state = init_state.entur.scripts.get_initial_state("test_data", "0900-entur-snapshot.csv", "Scooter",
                                                       number_of_scooters = 150, number_of_clusters = 5,
                                                       number_of_vans = 1, random_seed = 1)
    # state = init_state.cityBike.parse.get_initial_state(city="Oslo", week=WEEK, bike_class="Bike",
    #                                                     number_of_vans=1, random_seed=1)

    ###############################################################################
    # calculate ideal state

    # ideal_state = ideal_state.evenly_distributed_ideal_state(state)
    ideal_state = ideal_state.outflow_ideal_state(state)

    state.set_ideal_state(ideal_state)

    ###############################################################################

    # Set up policy

    policy = policies.DoNothing()
    # policy = policies.RandomActionPolicy()
    # policy = policies.RebalancingPolicy()
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False)
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True)

    ###############################################################################

    # Set up simulator
    simulator = sim.Simulator(
        get_time(day=7),
        policy,
        state,
        verbose=True,
        start_time = get_time(day=2, hour=8),
    )

    ###############################################################################

    # Run simulator
    simulator.run()

    ###############################################################################

    # Visualize results
    visualization.visualize_lost_demand([simulator], title=("Week " + str(WEEK)), week=WEEK)
