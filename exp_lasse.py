#!/bin/python3
"""
FOMO simulator main program
"""
import copy

import settings
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike

import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen

from progress.bar import Bar

import output
import target_state
import matplotlib.pyplot as plt

from helpers import *

###############################################################################

# Duration of each simulation run
DURATION = timeInMinutes(hours=24)

# Enter instance definition here.  For numbikes and numstations, enter 'None' to use dataset default
instances = [
    # Name,         URL,                                                          numbikes, numstations, week, day, hour
    ("Oslo",        "https://data.urbansharing.com/oslobysykkel.no/trips/v1/",        2000,        None,   33,   0,    6 ),
    # ("Bergen",      "https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",      None,        None,   33,   0,    6 ),
    #("Trondheim",   "https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",   800,        None,   28,   1,    6 ),
    # ("Oslo-vinter", "https://data.urbansharing.com/oslovintersykkel.no/trips/v1/",      400,        None,    7,   0,    6 ),
    # ("Edinburgh",   "https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/",  200,        None,   20,   0,    6 ),
]

# Enter analysis definition here
analyses = [
    # Name,        target_state,                                 policy,                  numvehicles   
    ("do_nothing", target_state.evenly_distributed_target_state, policies.DoNothing(),              1),
    ("random-1", target_state.evenly_distributed_target_state, policies.RandomActionPolicy(),         1),
    ("evenly-1",     target_state.evenly_distributed_target_state, policies.GreedyPolicy(),           1),
    ("outflow-1",    target_state.outflow_target_state,            policies.GreedyPolicy(),           1),
    ("equalprob-1",  target_state.equal_prob_target_state,         policies.GreedyPolicy(),           1),
    ("evenly-2",     target_state.evenly_distributed_target_state, policies.GreedyPolicy(),           2),
    ("outflow-2",    target_state.outflow_target_state,            policies.GreedyPolicy(),           2),
    ("equalprob-2",  target_state.equal_prob_target_state,         policies.GreedyPolicy(),           2),
]        

seeds = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
# seeds = [ 0] 
seeds=[]
for s in range(30):
    seeds.append(s) 

###############################################################################

def lostTripsPlot(cities, policies, starv, serr, cong, cerr):
    fig, plots = plt.subplots(nrows=1, ncols=len(cities), sharey=True)
    subPlots = [] # fix that subPlots returns a list only when more than one plot. Tried squeezy=False but it did not work for this purpose 
    if len(cities) == 1:
        subPlots.append(plots)
    else:
        for p in plots:
            subPlots.append(p)

    fig.suptitle("FOMO simulator - lost trips results\nImprovement from baseline (left bar) in % ", fontsize=15)
    w = 0.3
    pos = []
    for city in range(len(cities)):
        pos.append([])
        for i in range(len(policies)): # IMPROVED by more readable code here
            pos[city].append(starv[city][i] + cong[city][i])
        baseline = starv[city][0] + cong[city][0] # first policy is always baseline
        
        policyLabels = [] # fix policy labels
        for i in range(len(policies)):
            if i > 0:
                improved =  ( ((starv[city][i] + cong[city][i]) - baseline)/baseline)*100.0
                policyLabels.append(policies[i] + "\n(" + "{:.1f}".format(improved) + "%)")
            else:
                policyLabels.append(policies[i]) # label for baseline

        subPlots[city].bar(policyLabels, starv[city], w, label='Starvation')
        subPlots[city].errorbar(policyLabels, starv[city], yerr = serr[city], fmt='none', ecolor='black')
        subPlots[city].bar(policyLabels, cong[city], w, bottom=starv[city], label='Congestion')

        delta = 0.03 # skew the upper error-bar horisontally with delta to avoid that they can overwrite each other
        policiesPlussDelta = []
        for i in range(len(policies)):
            policiesPlussDelta.append(i + delta)
        subPlots[city].errorbar(policiesPlussDelta, pos[city], yerr= cerr[city], fmt='none', ecolor='black')
        subPlots[city].set_xlabel(cities[city])
        subPlots[city].set_ylabel("Violations (% of total number of trips)")
        subPlots[city].legend()

    plt.show()
###############################################################################

if __name__ == "__main__":

    starvations = []
    congestions = []

    starvations_stdev = []
    congestions_stdev = []

    for instance in instances:
        print("  instance: ", instance[0])

        starvations.append([])
        congestions.append([])

        starvations_stdev.append([])
        congestions_stdev.append([])

        for analysis in analyses:
            print("    analysis: ", analysis[0])

            if instance[0] == "Oslo":
                initial_state = init_state.get_initial_state(source=init_state.cityBike, url=instance[1], week=instance[4],
                                                        fromInclude=[2020, 7], toInclude= [2022,8],
                                                        random_seed=0, number_of_stations=instance[3], number_of_bikes=instance[2],
                                                        target_state=analysis[1])
            elif instance[0] == "Oslo-vinter":
                initial_state = init_state.get_initial_state(source=init_state.cityBike, url=instance[1], week=instance[4],
                                                        fromInclude=[2018, 12], toInclude= [2019, 3],
                                                        random_seed=0, number_of_stations=instance[3], number_of_bikes=instance[2],
                                                        target_state=analysis[1])
                                                        
            elif instance[0] == "Bergen":
                initial_state = init_state.get_initial_state(source=init_state.cityBike, url=instance[1], week=instance[4],
                                                        fromInclude=[2018, 9], toInclude= [2021, 9],
                                                        random_seed=0, number_of_stations=instance[3], number_of_bikes=instance[2],
                                                        target_state=analysis[1])        

            elif instance[0] == "Trondheim":
                initial_state = init_state.get_initial_state(source=init_state.cityBike, url=instance[1], week=instance[4],
                                                        fromInclude=[2018,9], toInclude= [2021, 9],
                                                        random_seed=0, number_of_stations=instance[3], number_of_bikes=instance[2],
                                                        target_state=analysis[1])                                         
            elif instance[0] == "Edinburgh":
                initial_state = init_state.get_initial_state(source=init_state.cityBike, url=instance[1], week=instance[4],
                                                        fromInclude=[2018,9], toInclude= [2021, 9],
                                                        random_seed=0, number_of_stations=instance[3], number_of_bikes=instance[2],
                                                        target_state=analysis[1])
            else:
                initial_state = init_state.get_initial_state(source=init_state.cityBike, url=instance[1], week=instance[4],
                                                         random_seed=0, number_of_stations=instance[3], number_of_bikes=instance[2],
                                                         target_state=analysis[1])

            simulations = []

            for seed in seeds:
                print("      seed: ", seed)

                state_copy = copy.deepcopy(initial_state)
                state_copy.set_seed(seed)
                state_copy.set_num_vehicles(analysis[3])

                simul = sim.Simulator(
                    initial_state = state_copy,
                    policy = analysis[2],
                    start_time = timeInMinutes(days=instance[5], hours=instance[6]),
                    duration = DURATION,
                    verbose = True,
                )
                
                simul.run()

                simulations.append(simul)

            metric = sim.Metric.merge_metrics([sim.metrics for sim in simulations])

            scale = 100 / metric.get_aggregate_value("trips")

            starvations[-1].append(scale * metric.get_aggregate_value("starvation"))
            congestions[-1].append(scale * metric.get_aggregate_value("congestion"))

            starvations_stdev[-1].append(scale * metric.get_aggregate_value("starvation_stdev"))
            congestions_stdev[-1].append(scale * metric.get_aggregate_value("congestion_stdev"))

    ###############################################################################

    instance_names = [ instance[0] for instance in instances ]
    analysis_names = [ analysis[0] for analysis in analyses ]

    lostTripsPlot(instance_names, analysis_names, starvations, starvations_stdev, congestions, congestions_stdev)

    plt.show()

    print(" bye bye")
