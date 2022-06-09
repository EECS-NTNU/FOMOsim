#!/bin/python3
"""
FOMO simulator main program
"""

import settings
import init_state
import init_state.fosen_haldorsen
import target_state
import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
#import policies.gleditsch_hagen
import sim
import visualization
from GUI.dashboard import GUI_main

def get_time(day=0, hour=0, minute=0):
    return 24*60*day + 60*hour + minute

duration = get_time(hour=4)

if settings.USER_INTERFACE_MODE == "CMD" or not GUI_main():

    WEEK = 30

    ###############################################################################
    # get initial state

    # state = init_state.entur.scripts.get_initial_state("test_data", "0900-entur-snapshot.csv", "Scooter",
    #                                                    number_of_scooters = 150, number_of_clusters = 5,
    #                                                    number_of_vans = 1, random_seed = 1)
    
    # state = init_state.cityBike.parse.get_initial_state(city="Oslo", week=WEEK, bike_class="Bike",
    #                                                      number_of_vans=1, random_seed=1)

    state = init_state.fosen_haldorsen.get_initial_state(init_hour=7, number_of_vans=1, random_seed=1)

    ###############################################################################
    # calculate target state

    # target_state = target_state.evenly_distributed_target_state(state)
    # target_state = target_state.outflow_target_state(state)
    # target_state = target_state.us_target_state(state)
    target_state = target_state.fosen_haldorsen_target_state(state)

    state.set_target_state(target_state)

    ###############################################################################
    # Set up policy

    # policy = policies.DoNothing()
    # policy = policies.RandomActionPolicy()
    # policy = policies.RebalancingPolicy()
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True)
    policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False)

    ###############################################################################
    # Set up simulator

    # for st in state.stations:
    #     print(f"{st.original_id:4}: {st.capacity:2}: {len(st.scooters):3}: {st.get_leave_intensity(0, 7)} ", end="")
    #     for hour in range(24):
    #         print(f"{st.get_target_state(0, hour)}, ", end="")
    #     print()

    simulator = sim.Simulator(
        initial_state = state,
        policy = policy,
        start_time = get_time(day=0, hour=7),
        duration = duration,
        verbose = True,
    )

    ###############################################################################
    # Run simulator

    simulator.run()

    ###############################################################################
    # Print some statistics

    print(f"Simulation time = {duration} minutes")
    print(f"Total requested trips = {simulator.metrics.get_aggregate_value('trips')}")
    print(f"Starvations = {simulator.metrics.get_aggregate_value('lost_demand')}")
    print(f"Congestions = {simulator.metrics.get_aggregate_value('congestion')}")

    ###############################################################################
    # Visualize results

    # visualization.visualize_trips([simulator], title=("Week " + str(WEEK)), week=WEEK)
    # visualization.visualize_starvation([simulator], title=("Week " + str(WEEK)), week=WEEK)
    # visualization.visualize_congestion([simulator], title=("Week " + str(WEEK)), week=WEEK)
