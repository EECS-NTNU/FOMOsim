#!/bin/python3

import itertools as it
import os

import settings
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike
import target_state
import policies
import policies.fosen_haldorsen
import policies.gleditsch_hagen
import sim
import output
from helpers import timeInMinutes

simple_run = True

# BASE DATA

# Parameters
start_hour = 9   #7*60 = 420
simulation_time = 120  # 7 am to 11 pm   = 60*16=960   -> Smaller: 240 (60*4)
num_stations = 15   #was at 200
num_vehicles = 1
subproblem_scenarios = 2   #was at ten
branching = 7
time_horizon=30   

#basic_seed = 1   #alternatively just do a seed here at the beginning. A bit less controll though. 
seed_generating_trips = 1
seed_scenarios_subproblems = 2    # TO DO

greedy = False
fh = False
gh = True

# SCENARIO DATA

seeds_generating_trips = list(range(0,10))
inputs = {
    'greedy':[False],
    #'branching':[3,5,7],
    #'num_vehicles':[0],
    'num_stations':[50],
}


if simple_run:

    #Make sure the working directory is in the right place, otherwise it cannot find things
    #os.chdir("C:\\Users\\steff\\OneDrive - NTNU\\Work\\GitHub\\FOMO-sim\\fomo")

    # FOSEN HALDORSEN
    
    tstate = target_state.us_target_state
    state = init_state.get_initial_state(source=init_state.fosen_haldorsen,
                                        target_state=tstate,
                                        init_hour=start_hour, 
                                        number_of_stations=num_stations,
                                        number_of_vehicles=num_vehicles)


    #CITY BIKES

    # START_TIME = timeInMinutes(hours=7)
    # DURATION = timeInMinutes(minutes=120, hours=0)
    # WEEK = 12
    
    #tstate = target_state.equal_prob_target_state
    # state = init_state.get_initial_state(source=init_state.cityBike,
    #                                      target_state=tstate,
    #                                      url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/", 
    #                                      week=WEEK, bike_class="Bike", 
    #                                      number_of_vehicles=1, 
    #                                      number_of_stations=15)
    
    #state.set_seed(seed_generating_trips)

    # setup policy
    if greedy:
        policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True)
    elif fh:
        policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False, scenarios=subproblem_scenarios,
                                                               branching=branching, time_horizon=time_horizon)
    elif gh:
        policy = policies.gleditsch_hagen.GleditschHagenPolicy(variant='PatternBased')    


    # setup simulator
    simulator = sim.Simulator(
        initial_state = state,
        policy = policy,
        start_time = timeInMinutes(hours=start_hour),
        duration = simulation_time+1,
        verbose = True,
    )

    # run simulator
    simulator.run()
    #mainly interested in world.policy.get_best_action(world, self.vehicle)


    # results
    print(f"Simulation time = {simulation_time} minutes")
    print(f"Total requested trips = {simulator.metrics.get_aggregate_value('trips')}")
    print(f"Starvations = {simulator.metrics.get_aggregate_value('lost_demand')}")
    print(f"Congestions = {simulator.metrics.get_aggregate_value('congestion')}")

    output.write_csv(simulator, "output.csv", hourly = True)


else:

    keys = sorted(inputs)
    combinations = list(it.product(*(inputs[key] for key in keys)))
    num_scenario_analyses = len(combinations)

    if os.path.exists("output.csv"):
        os.remove("output.csv")

    for i in range(num_scenario_analyses):
        simulators = []

        # set scenario parameters
        values = combinations[i]
        parameters = dict(zip(keys, values))
        for key, value in parameters.items():
            locals()[key] = value

        for seed_generating_trips in seeds_generating_trips:
            print("--------------------- SIMULATION ------------------------------")

            # print scenario parameters
            print("seed_generating_trips", seed_generating_trips)
            for key, value in parameters.items():
                print(key, value)

            # setup state
            tstate = target_state.us_target_state
            state = init_state.get_initial_state(source=init_state.fosen_haldorsen,
                                                 target_state=tstate,
                                                 init_hour=start_hour, number_of_stations=num_stations,
                                                 number_of_vehicles=num_vehicles)
            state.set_seed(seed_generating_trips)

            # setup policy
            if greedy:
                policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True)
            else:
                policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False, scenarios=subproblem_scenarios,
                                                                       branching=branching, time_horizon=time_horizon)

            # setup simulator
            simulators.append(sim.Simulator(
                initial_state = state,
                policy = policy,
                start_time = timeInMinutes(hours=start_hour),
                duration = simulation_time+1,
                verbose = True,
            ))

            # run simulator
            simulators[-1].run()

            # results
            print(f"Simulation time = {simulation_time} minutes")
            print(f"Total requested trips = {simulators[-1].metrics.get_aggregate_value('trips')}")
            print(f"Starvations = {simulators[-1].metrics.get_aggregate_value('lost_demand')}")
            print(f"Congestions = {simulators[-1].metrics.get_aggregate_value('congestion')}")
            print()

        output.write_csv(simulators, "output.csv", hourly = True, mode="a", parameters=parameters.items())
