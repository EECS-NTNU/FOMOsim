# !/bin/python3
"""
FOMO simulator main program
"""

import copy

import settings
import sim
import clustering.scripts

import policies
import policies.fosen_haldorsen
from visualization.visualizer import visualize_analysis
import ideal_state


from GUI.dashboard import GUI_main

PERIOD = 960  # 16 hours

simulators = []

###############################################################################
# Set up initial state

if settings.USER_INTERFACE_MODE == "CMD" or not GUI_main():

    # state = get_initial_state(city = "Oslo", week=12)

    # This is frm Haflan Haga and Spetalen
    state = clustering.scripts.get_initial_state(
        "test_data",
        "0900-entur-snapshot.csv",
        "Scooter",
        number_of_scooters = 2500,
        number_of_clusters = 10,
        number_of_vans = 2,
        random_seed = 1,
    )

    ###############################################################################
    # calculate ideal state

    # ideal_state = ideal_state.evenly_distributed_ideal_state(state)
    # state.set_ideal_state(ideal_state)
    ideal_state.haflan_haga_spetalen_ideal_state(state)

    ###############################################################################

    # # Set up simulator
    # simulators.append(sim.Simulator(
    #     PERIOD,
    #     policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False),
    #     copy.deepcopy(state),
    #     verbose=True,
    #     label="FosenHaldorsen",
    # ))

    # # Run first simulator
    # simulators[-1].run()

    ###############################################################################

    # # Set up simulator
    # simulators.append(sim.Simulator(
    #     PERIOD,
    #     policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True),
    #     copy.deepcopy(state),
    #     verbose=True,
    #     label="Greedy",
    # ))

    # # Run first simulator
    # simulators[-1].run()

    ###############################################################################

    # Set up simulator
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

    # Set up simulator
    simulators.append(sim.Simulator(
        PERIOD,
        policies.RebalancingPolicy(),
        copy.deepcopy(state),
        verbose=True,
        label="Rebalancing",
    ))

    # Run first simulator
    simulators[-1].run()

    ###############################################################################

    # Visualize results
    visualize_analysis(simulators)
