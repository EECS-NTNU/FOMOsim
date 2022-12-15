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
import policies.inngjerdingen_moeller.inngjerdingen_moeller
import sim
import output
import demand
from helpers import timeInMinutes

from output.plots import cityTrafficStats

START_TIME = timeInMinutes(hours=6)
DURATION = timeInMinutes(hours=16)
# DURATION = timeInMinutes(hours=2)
INSTANCE = 'TD_W34'
WEEK = 34

def main():

    ###############################################################################
    # Get initial state

    # the following is for creating a new initial state from trip data
    # state = init_state.get_initial_state(name="Oslo",
    #                                      source=init_state.cityBike,
    #                                      number_of_stations=None,
    #                                      number_of_bikes=2000,
    #                                      mapdata=("instances/oslo.png", (10.6365, 10.8631, 59.8843, 59.9569)),
    #                                      url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",
    #                                      week=31)

    # the following is for reading a precalculated initial state from a json file
    state = init_state.read_initial_state("instances/"+INSTANCE);

    state.set_seed(1)

    ###############################################################################
    # Set up vehicles
    # Each vehicle has an associated policy

    # policy = policies.RandomActionPolicy()
    # policy = policies.GreedyPolicy()
    policy = policies.inngjerdingen_moeller.inngjerdingen_moeller.InngjerdingenMoellerPolicy(time_horizon=25)
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True)
    # policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False, scenarios=2, branching=7, time_horizon=25)
    # policy = policies.gleditsch_hagen.GleditschHagenPolicy(variant='PatternBased')
    
    state.set_vehicles([policy]) # this creates one vehicle for each policy in the list

    ###############################################################################
    # Set up target state

    # tstate = target_state.EvenlyDistributedTargetState()
    # tstate = target_state.OutflowTargetState()
    # tstate = target_state.EqualProbTargetState()
    tstate = target_state.USTargetState()
    # tstate = target_state.HalfCapacityTargetState()

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

    #If comparissons between roaming=True and roaming=False: 
    print(f"Different station choices = {simulator.metrics.get_aggregate_value('Different station choice')}")
    print(f"Different pickup quantities = {simulator.metrics.get_aggregate_value('Different pickup quantity')}")
    print(f"Different deliver quantities = {simulator.metrics.get_aggregate_value('Different delivery quantity')}")
    print(f"Number of overlaps = {simulator.metrics.get_aggregate_value('Overlap')}")
    print(f"Number of identical choices = {simulator.metrics.get_aggregate_value('Same choice')}")
    print(f"Number of subproblems solved = {simulator.metrics.get_aggregate_value('Number of subproblems')}")
    print(f"Number of different choices = {simulator.metrics.get_aggregate_value('Different choice')}")
    
    # Output to file

    output.write_csv(simulator, "output.csv", hourly = False)

    # Plot to screen
    output.visualize([simulator.metrics], metric="trips")
    output.visualize([simulator.metrics], metric="starvation")
    output.visualize([simulator.metrics], metric="congestion")
    output.visualize_heatmap([simulator], metric="trips")

#If comparissons between roaming=True and roaming=False : 
    output.visualize([simulator.metrics], metric="Different station choice")
    output.visualize([simulator.metrics], metric="Different pickup quantity")
    output.visualize([simulator.metrics], metric="Different delivery quantity")
    output.visualize([simulator.metrics], metric="Number of subproblems")
    output.visualize([simulator.metrics], metric="Different choice")
    output.visualize([simulator.metrics], metric="Same choice")
    output.visualize([simulator.metrics], metric="Overlap")
    
    
# show travel times for a given bike
    bikes = simulator.state.get_all_bikes()
    bikes = sorted(bikes, key=lambda bike: bike.metrics.getLen("travel_time"), reverse=True)
    print(f"Bike {bikes[11].id}: {bikes[11].metrics.getSum('travel_time')} {bikes[11].metrics.getSum('travel_time_congested')}")
    output.visualize([bikes[11].metrics], metric="travel_time")
    output.visualize([bikes[11].metrics], metric="travel_time_congested")


if __name__ == "__main__":
    main() 

