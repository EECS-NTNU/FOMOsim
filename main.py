#!/bin/python3
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
import policies.haflan_haga_spetalen
import policies.gleditsch_hagen
from visualization.visualizer import visualize_analysis
import ideal_state

from GUI.dashboard import GUI_main
from tripStats.helpers import dateAndTimeStr

simulators = []

def get_time(day=0, hour=0, minute=0):
    return 24*60*day + 60*hour + minute

WEEK = 30
START_DAY = 2
START_HOUR = 8
PERIOD = get_time(4)

###############################################################################
# Set up initial state

if settings.USER_INTERFACE_MODE == "CMD" or not GUI_main():

    ###############################################################################
    # get initial state

    #state, _ = tripStats.parse.get_initial_state(city="Oslo", week=WEEK)
    state = clustering.scripts.get_initial_state("test_data", "0900-entur-snapshot.csv", "Scooter", number_of_scooters = 500, number_of_clusters = 10, number_of_vans = 1, random_seed = 1)

    ###############################################################################
    # calculate ideal state

    ideal_state = ideal_state.evenly_distributed_ideal_state(state)
    #ideal_state = ideal_state.outflow_ideal_state(state)
    state.set_ideal_state(ideal_state)

    ###############################################################################

    # # Set up simulator
    # simul = sim.Simulator.load("sim_cache/entur_scooter_10_500.pickle")

    # hhsstate = copy.deepcopy(state)
    # hhsstate.simulation_scenarios = simul.state.simulation_scenarios

    # simul.init(
    #     PERIOD, 
    #     hhsstate,
    #     verbose=True,
    #     start_time = get_time(day=START_DAY, hour=START_HOUR),
    #     label="HHS",
    # )
    # simulators.append(simul)

    # # Run simulator
    # simulators[-1].run()

    # ###############################################################################

    # # Set up simulator
    # simulators.append(sim.Simulator(
    #     PERIOD,
    #     policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False),
    #     copy.deepcopy(state),
    #     verbose=True,
    #     start_time = get_time(day=START_DAY, hour=START_HOUR),
    #     label="FH",
    # ))

    # # Run simulator
    # simulators[-1].run()

    # ###############################################################################

    # Set up simulator
    simulators.append(sim.Simulator(
        PERIOD,
        policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True),
        copy.deepcopy(state),
        verbose=True,
        start_time = get_time(day=START_DAY, hour=START_HOUR),
        label="FH_Greedy",
    ))

    # Run simulator
    simulators[-1].run()

    # ###############################################################################

    # Set up simulator
    simulators.append(sim.Simulator(
        PERIOD,
        policies.DoNothing(),
        copy.deepcopy(state),
        verbose=True,
        start_time = get_time(day=START_DAY, hour=START_HOUR),
        label="DoNothing",
    ))

    # Run simulator
    simulators[-1].run()

    # ###############################################################################

    # Set up simulator
    simulators.append(sim.Simulator(
        PERIOD,
        policies.RebalancingPolicy(),
        copy.deepcopy(state),
        verbose=True,
        start_time = get_time(day=START_DAY, hour=START_HOUR),
        label="Rebalancing",
    ))

    # Run simulator
    simulators[-1].run()

    # ###############################################################################

    # Set up simulator
    simulators.append(sim.Simulator(
        PERIOD,
        policies.RebalancingPolicy(),
        copy.deepcopy(state),
        verbose=True,
        start_time = get_time(day=START_DAY, hour=START_HOUR),
        label="Rebalancing",
    ))

    # Run simulator
    simulators[-1].run()

    # ###############################################################################

    # # Set up simulator
    # simulators.append(sim.Simulator(
    #     PERIOD,
    #     policies.gleditsch_hagen.GleditschHagenPolicy(),
    #     copy.deepcopy(state),
    #     verbose=True,
    #     start_time = get_time(day=START_DAY, hour=START_HOUR),
    #     label="GH",
    # ))

    # # Run simulator
    # simulators[-1].run()

    ###############################################################################

    # Visualize results
    visualize_analysis(simulators)
