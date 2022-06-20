#!/bin/python3
"""
FOMO simulator main program
"""
import copy

import settings
import sim
import init_state
import init_state.fosen_haldorsen

import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import policies.gleditsch_hagen

from progress.bar import Bar

import output
import target_state

from GUI.dashboard import GUI_main
from init_state.cityBike.helpers import dateAndTimeStr

def get_time(day=0, hour=0, minute=0):
    return 24*60*day + 60*hour + minute

#WEEK = 1
#WEEK = 28
WEEK = 33
START_DAY = 0
START_HOUR = 7
PERIOD = get_time(hour=48)
RUNS = 10

def run_sim(state, period, policy, start_time, label, seed):
    local_state = copy.deepcopy(state)
    local_state.set_seed(seed)

    # Set up simulator
    simul = sim.Simulator(
        initial_state = local_state,
        policy = policy, 
        start_time = start_time,
        duration = PERIOD,
        verbose = True,
        label = label,
    )

    # Run simulator
    simul.run()
    return simul

if settings.USER_INTERFACE_MODE == "CMD" or not GUI_main():

    start_time = get_time(day=START_DAY, hour=START_HOUR)

    ###############################################################################
    # get initial state

    # state = init_state.entur.scripts.get_initial_state("test_data", "0900-entur-snapshot.csv", "Scooter",
    #                                                    number_of_scooters = 300, number_of_clusters = 50,
    #                                                    number_of_vans = 3, random_seed = 1)

    # state = init_state.cityBike.parse.get_initial_state(city="Oslo", week=WEEK, bike_class="Bike",
    #                                                     number_of_vans=3, random_seed=1)

    state = init_state.fosen_haldorsen.get_initial_state(init_hour=start_time//60, number_of_stations=50, number_of_vans=3, random_seed=1)

    ###############################################################################
    # calculate target state

    tstate = target_state.us_target_state(state)
    #tstate = target_state.equal_prob_target_state(state)
    state.set_target_state(tstate)

    ###############################################################################

    # simulations = []

    # xvalues = [0, 1, 2, 4, 8]

    # progress = Bar(
    #     "Running",
    #     max = len(xvalues) * RUNS,
    # )

    # for num_vans in xvalues:
    #     state.set_num_vans(num_vans)

    #     sims = []
    
    #     for run in range(RUNS):
    #         #sims.append(run_sim(state, PERIOD, policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True), start_time, "FH-Greedy", run))
    #         sims.append(run_sim(state, PERIOD, policies.GreedyPolicy(), start_time, "HHS-Greedy", run))
    #         progress.next()

    #     simulations.append(sims)
        
    # progress.finish()

    # # Visualize results
    # visualize_end(simulations, xvalues, title=("Week " + str(WEEK)), week=WEEK)

    ###############################################################################

    # Set up simulator
    # simul = sim.Simulator.load("sim_cache/fh_3_50.pickle")

    # hhs = []

    # for run in range(RUNS):
    #     print(f"\nRun {run+1} of {RUNS}")

    #     hhsstate = copy.deepcopy(state)
    #     hhsstate.simulation_scenarios = simul.state.simulation_scenarios

    #     hhsstate.set_seed(run)

    #     simul.init(
    #         duration = PERIOD, 
    #         initial_state = hhsstate,
    #         verbose = True,
    #         start_time = start_time, 
    #         label = "HHS",
    #     )
    #     simul.run()

    #     hhs.append(simul)

    #     # Run simulator

    # output.write_csv(hhs, "output_hhs.csv", WEEK, hourly = True)

    ###############################################################################

    # us = []
    # equalprob = []
    # outflow = []
    # even = []

    # tstate = target_state.fosen_haldorsen_target_state(state)
    # state.set_target_state(tstate)
    # for run in range(RUNS):
    #     print(f"\nRun {run+1} of {RUNS*4}")
    #     us.append (run_sim(state, PERIOD, policies.RebalancingPolicy(), start_time, "US",  run))
    # output.write_csv(us, "output_us.csv", WEEK, hourly = True)

    # tstate = target_state.us_target_state(state)
    # state.set_target_state(tstate)
    # for run in range(RUNS):
    #     print(f"\nRun {run+11} of {RUNS*4}")
    #     equalprob.append (run_sim(state, PERIOD, policies.RebalancingPolicy(), start_time, "EqualProb",  run))
    # output.write_csv(equalprob, "output_same.csv", WEEK, hourly = True)

    # tstate = target_state.outflow_target_state(state)
    # state.set_target_state(tstate)
    # for run in range(RUNS):
    #     print(f"\nRun {run+21} of {RUNS*4}")
    #     outflow.append (run_sim(state, PERIOD, policies.RebalancingPolicy(), start_time, "Outflow",  run))
    # output.write_csv(outflow, "output_outflow.csv", WEEK, hourly = True)

    # tstate = target_state.evenly_distributed_target_state(state)
    # state.set_target_state(tstate)
    # for run in range(RUNS):
    #     print(f"\nRun {run+31} of {RUNS*4}")
    #     even.append (run_sim(state, PERIOD, policies.RebalancingPolicy(), start_time, "Even",  run))
    # output.write_csv(even, "output_even.csv", WEEK, hourly = True)

    ###############################################################################

    donothing = []
    random = []
    greedy = []
    fhgreedy = []
    fh = []

    for run in range(RUNS):
        print(f"\nRun {run+1} of {RUNS}")
        donothing.append (run_sim(state, PERIOD, policies.DoNothing(), start_time, "DoNothing",  run))
        #random.append (run_sim(state, PERIOD, policies.RandomActionPolicy(), start_time, "Random",  run))
        #greedy.append (run_sim(state, PERIOD, policies.GreedyPolicy(), start_time, "Greedy",  run))
        #fhgreedy.append (run_sim(state, PERIOD, policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True), start_time, "FH-greedy",  run))
        #fh.append (run_sim(state, PERIOD, policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False, scenarios=2, branching=7, time_horizon=25), start_time, "FH",  run))

    output.write_csv(donothing, "output_donothing.csv", WEEK, hourly = True)
    #output.write_csv(random, "output_random.csv", WEEK, hourly = True)
    #output.write_csv(greedy, "output_greedy.csv", WEEK, hourly = True)
    #output.write_csv(fhgreedy, "output_fhgreedy.csv", WEEK, hourly = True)
    #output.write_csv(fh, "output_fh.csv", WEEK, hourly = True)

    #output.visualize_starvation([donothing, random, greedy, fhgreedy], title=("Week " + str(WEEK)), week=WEEK)
    #output.visualize_congestion([donothing, random, greedy, fhgreedy], title=("Week " + str(WEEK)), week=WEEK)
