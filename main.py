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
import output
from helpers import timeInMinutes
from GUI.dashboard import GUI_main

start_time = timeInMinutes(hour=7)
duration = timeInMinutes(hour=48)

if settings.USER_INTERFACE_MODE == "CMD" or not GUI_main():

    WEEK = 12

    ###############################################################################
    # get initial state

    #cityURL = "https://data.urbansharing.com/oslobysykkel.no/trips/v1/"
    cityURL = "https://data.urbansharing.com/oslovintersykkel.no/trips/v1/"
    #cityURL = "https://data.urbansharing.com/bergenbysykkel.no/trips/v1/"
    #cityURL = "https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/"
    #cityURL = "https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/"
    state = init_state.cityBike.parse.get_initial_state(url=cityURL,  week=WEEK, bike_class="Bike", number_of_vehicles=3, random_seed=1)

    # state = init_state.entur.scripts.get_initial_state("test_data", "0900-entur-snapshot.csv", "Scooter",
    #                                                    number_of_scooters = 150, number_of_clusters = 5,
    #                                                    number_of_vehicles = 3, random_seed = 1)
    
    # state = init_state.fosen_haldorsen.get_initial_state(init_hour=start_time//60, number_of_stations=50,
    #                                                      number_of_vehicles=3, random_seed=1)

    ###############################################################################
    # calculate target state


    # tstate = target_state.evenly_distributed_target_state(state)
    # tstate = target_state.outflow_target_state(state)
    tstate = target_state.equal_prob_target_state(state)

    state.set_target_state(tstate)

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
        start_time = start_time,
        duration = duration,
        verbose = True,
    )

    ###############################################################################
    # Run simulator

    simulator.run()

    ###############################################################################
    # Output

    print(f"Simulation time = {duration} minutes")
    print(f"Total requested trips = {simulator.metrics.get_aggregate_value('trips')}")
    print(f"Starvations = {simulator.metrics.get_aggregate_value('lost_demand')}")
    print(f"Congestions = {simulator.metrics.get_aggregate_value('congestion')}")

    output.write_csv(simulator, "output.csv", WEEK, hourly = False)

    output.visualize_trips([simulator], title=("Week " + str(WEEK)), week=WEEK)
    output.visualize_starvation([simulator], title=("Week " + str(WEEK)), week=WEEK)
    output.visualize_congestion([simulator], title=("Week " + str(WEEK)), week=WEEK)
