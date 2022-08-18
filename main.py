#!/bin/python3
"""
FOMO simulator example
"""

import settings
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike
import target_state
import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
#import policies.gleditsch_hagen
import sim
import output
from helpers import timeInMinutes
#   from GUI.dashboard import GUI_main

START_TIME = timeInMinutes(hours=7)
DURATION = timeInMinutes(days=1)
WEEK = 30

def main():

    ###############################################################################
    # Get initial state

    # tstate = target_state.evenly_distributed_target_state
    # tstate = target_state.outflow_target_state
    tstate = target_state.equal_prob_target_state

    state = init_state.get_initial_state(source=init_state.cityBike,
                                         url="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",
                                         week=WEEK, random_seed=0,
                                         target_state=tstate,
                                         )

    # state = init_state.get_initial_state(source=init_state.entur,
    #                                      target_state=tstate,
    #                                      entur_data_dir="test_data", entur_main_file="0900-entur-snapshot.csv", 
    #                                      number_of_bikes = 150, number_of_clusters = 5, number_of_vehicles = 3, random_seed = 1)
    
    # state = init_state.get_initial_state(source=init_state.fosen_haldorsen,
    #                                      target_state=tstate,
    #                                      init_hour=START_TIME//60, number_of_stations=50, number_of_vehicles=3, random_seed=1)

    ###############################################################################
    # Set up policy

    policy = policies.DoNothing()
    # policy = policies.RandomActionPolicy()
    # policy = policies.GreedyPolicy()
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True)
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False, scenarios=2, branching=7, time_horizon=25)

    ###############################################################################
    # Set up simulator

    simulator = sim.Simulator(
        initial_state = state,
        policy = policy,
        start_time = START_TIME,
        duration = DURATION,
        verbose = True,
    )

    ###############################################################################
    # Run simulator

    simulator.run()

    ###############################################################################
    # Output

    print(f"Simulation time = {DURATION} minutes")
    print(f"Total requested trips = {simulator.metrics.get_aggregate_value('trips')}")
    print(f"Starvations = {simulator.metrics.get_aggregate_value('lost_demand')}")
    print(f"Congestions = {simulator.metrics.get_aggregate_value('congestion')}")

    output.write_csv(simulator, "output.csv", WEEK, hourly = False)

    output.visualize_trips([simulator], title=("Week " + str(WEEK)), week=WEEK)
    output.visualize_starvation([simulator], title=("Week " + str(WEEK)), week=WEEK)
    output.visualize_congestion([simulator], title=("Week " + str(WEEK)), week=WEEK)


if __name__ == "__main__":
    main()
