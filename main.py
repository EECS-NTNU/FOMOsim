#!/bin/python3
"""
FOMO simulator main program
"""
import copy

import settings
import sim
import init_state

import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import policies.gleditsch_hagen

from visualization.visualizer import visualize_analysis
import ideal_state

from GUI.dashboard import GUI_main
from init_state.cityBike.helpers import dateAndTimeStr

def get_time(day=0, hour=0, minute=0):
    return 24*60*day + 60*hour + minute

#WEEK = 28
WEEK = 33
START_DAY = 0
START_HOUR = 0
PERIOD = get_time(day=7)
RUNS = 10

def run_sim(state, period, policy, start_time, label, seed):
    local_state = copy.deepcopy(state)
    local_state.set_seed(seed)

    # Set up simulator
    simul = sim.Simulator(
        PERIOD,
        policy, 
        local_state,
        verbose=True,
        start_time = start_time,
        label=label,
    )

    # Run simulator
    simul.run()
    return simul

if settings.USER_INTERFACE_MODE == "CMD" or not GUI_main():

    start_time = get_time(day=START_DAY, hour=START_HOUR)

    ###############################################################################
    # get initial state

    #state = init_state.entur.scripts.get_initial_state("test_data", "0900-entur-snapshot.csv", "Bike", number_of_scooters = 250, number_of_clusters = 5, number_of_vans = 1, random_seed = 1)
    state = init_state.cityBike.parse.get_initial_state(city="Oslo", week=WEEK, bike_class="Bike", number_of_vans=1, random_seed=1)

    ###############################################################################
    # calculate ideal state

    #ideal_state = ideal_state.evenly_distributed_ideal_state(state)
    ideal_state = ideal_state.outflow_ideal_state(state)
    state.set_ideal_state(ideal_state)

    ###############################################################################

    donothings = []
    randoms = []
    rebalancings = []
    
    for run in range(RUNS):
        donothings.append  (run_sim(state, PERIOD, policies.DoNothing(),          start_time, "DoNothing",   run))
        randoms.append     (run_sim(state, PERIOD, policies.RandomActionPolicy(), start_time, "Random",      run))
        rebalancings.append(run_sim(state, PERIOD, policies.RebalancingPolicy(),  start_time, "Rebalancing", run))

    ##############################################################################

    num_vans = 2

    state.set_num_vans(num_vans)

    randoms2 = []
    rebalancings2 = []
    
    for run in range(RUNS):
        randoms2.append     (run_sim(state, PERIOD, policies.RandomActionPolicy(), start_time, "Random " + str(num_vans) + " vans",      run))
        rebalancings2.append(run_sim(state, PERIOD, policies.RebalancingPolicy(),  start_time, "Rebalancing " + str(num_vans) + " vans", run))

    ##############################################################################

    # Visualize results
    visualize_analysis([donothings, randoms, rebalancings, randoms2, rebalancings2], title=("Week " + str(WEEK)), week=WEEK)

