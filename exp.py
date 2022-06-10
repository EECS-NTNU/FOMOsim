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

from visualization.visualizer import visualize_end, visualize_starvation, visualize_congestion, save_lost_demand_csv
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
PERIOD = get_time(hour=4)
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
        verbose = False,
        label = label,
    )

    # Run simulator
    simul.run()
    return simul

if settings.USER_INTERFACE_MODE == "CMD" or not GUI_main():

    start_time = get_time(day=START_DAY, hour=START_HOUR)

    ###############################################################################
    # get initial state

    state = init_state.fosen_haldorsen.get_initial_state(init_hour=7, number_of_stations = 50, number_of_vans=3, random_seed=1)

    ###############################################################################
    # calculate target state

    tstate = target_state.fosen_haldorsen_target_state(state)
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
    #         sims.append(run_sim(state, PERIOD, policies.RebalancingPolicy(), start_time, "HHS-Greedy", run))
    #         progress.next()

    #     simulations.append(sims)
        
    # progress.finish()

    # # Visualize results
    # visualize_end(simulations, xvalues, title=("Week " + str(WEEK)), week=WEEK)

    ###############################################################################

    donothings = []
    rebalancings = []
    fhgreedys = []
    fhs = []
    
    state.set_num_vans(3)

    progress = Bar(
        "Running",
        max = RUNS,
    )

    for run in range(RUNS):
        donothings.append  (run_sim(state, PERIOD, policies.DoNothing(),                                       start_time, "DoNothing",  run))
        rebalancings.append (run_sim(state, PERIOD, policies.RebalancingPolicy(),                                       start_time, "FH-HHS",  run))
        fhgreedys.append   (run_sim(state, PERIOD, policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True), start_time, "FH-Greedy",  run))
        fhs.append         (run_sim(state, PERIOD, policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False, scenarios=2, branching=7, time_horizon=25), start_time, "FH",  run))

        progress.next()

    progress.finish()
        
    # Visualize results
    visualize_starvation([donothings, fhgreedys, fhs], title=("Week " + str(WEEK)), week=WEEK)
    visualize_congestion([donothings, fhgreedys, fhs], title=("Week " + str(WEEK)), week=WEEK)
