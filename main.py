#!/bin/python3
"""
FOMO simulator example
"""

from settings import *
import init_state
import init_state.cityBike
import target_state
import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import policies.gleditsch_hagen
import sim
import output
from helpers import timeInMinutes

from output.plots import cityTrafficStats

START_TIME = timeInMinutes(hours=7)
NUM_DAYS = 7
DURATION = timeInMinutes(hours=24*NUM_DAYS)
instance = 'OS_W31'

def main():

    ###############################################################################
    # Get initial state

    # tstate = target_state.evenly_distributed_target_state
    # tstate = target_state.outflow_target_state
    tstate = target_state.equal_prob_target_state

    #state = init_state.read_initial_state("instances/Oslo", tstate);
    state = init_state.read_initial_state("instances/"+instance, tstate);


    ###############################################################################
    # Set up policy

    # policy = policies.RandomActionPolicy()
    policy = policies.GreedyPolicy()
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True)
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False, scenarios=2, branching=7, time_horizon=25)
    # policy = policies.gleditsch_hagen.GleditschHagenPolicy(variant='PatternBased')
    
    state.set_vehicles([policy])

    ###############################################################################
    # Set up simulator

    simulator = sim.Simulator(
        initial_state = state,
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
    print(f"Starvations = {simulator.metrics.get_aggregate_value('starvation')}")
    print(f"Congestions = {simulator.metrics.get_aggregate_value('congestion')}")

    output.write_csv(simulator, "output.csv", hourly = False)

    # output.visualize_trips([simulator], title=("Week " + str(WEEK)), week=WEEK)
    # output.visualize_starvation([simulator], title=("Week " + str(WEEK)), week=WEEK)
    # output.visualize_congestion([simulator], title=("Week " + str(WEEK)), week=WEEK)

    output.visualize_heatmap(simulator)

if __name__ == "__main__":
    main()
