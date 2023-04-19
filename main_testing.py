#!/bin/python3
"""
FOMO simulator example
"""

from settings import *
import init_state
import init_state.json_source
import init_state.csv_source
import target_state
import policies
import policies.fosen_haldorsen
#import policies.haflan_haga_spetalen
import policies.gleditsch_hagen
import demand
import sim
import output
from helpers import timeInMinutes

from output.plots import cityTrafficStats

import cProfile
import pstats

START_TIME = timeInMinutes(hours=7)
DURATION = timeInMinutes(hours=1)
INSTANCE = 'TD_W34'

def main():

    ###############################################################################
    # Get initial state

    # the following is for creating a new initial state from trip data
    # state = init_state.get_initial_state(source=init_state.json_source,
    #                                      name="Trondheim",
    #                                      urlHistorical="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",
    #                                      urlGbfs="https://gbfs.urbansharing.com/trondheimbysykkel.no",
    #                                      week=34)

    # the following is for reading a precalculated initial state from a json file
    state = init_state.read_initial_state("instances/"+INSTANCE);

    state.set_seed(1)

    ###############################################################################
    # Set up vehicles
    # Each vehicle has an associated policy

    # policy = policies.RandomActionPolicy()
    # policy = policies.GreedyPolicy()
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True)
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False, scenarios=2, branching=7, time_horizon=25)
    policy = policies.gleditsch_hagen.GleditschHagenPolicy(variant='PatternBased')
    
    state.set_vehicles([policy]) # this creates one vehicle for each policy in the list

    ###############################################################################
    # Set up target state

    # tstate = target_state.EvenlyDistributedTargetState()
    # tstate = target_state.OutflowTargetState()
    tstate = target_state.EqualProbTargetState()
    # tstate = target_state.USTargetState()

    ###############################################################################
    # Set up demand

    dmand = demand.Demand()

    ###############################################################################
    # Set up simulator

    simulator = sim.Simulator(
        initial_state = state,
        target_state = tstate,
        demand = dmand,
        start_time = START_TIME,
        duration = DURATION,
        verbose = True,
    )
    simulator.run()

    # Output to console

    print(f"Simulation time = {DURATION} minutes")
    print(f"Total requested trips = {simulator.metrics.get_aggregate_value('trips')}")
    print(f"Starvations = {simulator.metrics.get_aggregate_value('starvation')}")
    print(f"Congestions = {simulator.metrics.get_aggregate_value('congestion')}")

    # Output to file

    output.write_csv(simulator, "output.csv", hourly = False)

    # # Plot to screen

    output.visualize([simulator.metrics], metric="trips")
    output.visualize([simulator.metrics], metric="starvation")
    output.visualize([simulator.metrics], metric="congestion")
    output.visualize_heatmap([simulator], metric="trips")

    # show travel times for a given bike
    bikes = simulator.state.get_all_bikes()
    bikes = sorted(bikes, key=lambda bike: bike.metrics.getLen("travel_time"), reverse=True)
    print(f"Bike {bikes[11].id}: {bikes[11].metrics.getSum('travel_time')} {bikes[11].metrics.getSum('travel_time_congested')}")
    output.visualize([bikes[11].metrics], metric="travel_time")
    output.visualize([bikes[11].metrics], metric="travel_time_congested")


if __name__ == "__main__":
    
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    profiler.dump_stats('restats')
    p = pstats.Stats('restats')
    p.sort_stats('time').print_stats(100) # print the stats after sorting by time
    p.sort_stats('cumtime').print_stats(100) # print the stats after sorting by time