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

cities = [
    "https://data.urbansharing.com/oslobysykkel.no/trips/v1/",
#    "https://data.urbansharing.com/oslovintersykkel.no/trips/v1/",
#    "https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",
#    "https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",
#    "https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/",
]

target_states = [
    target_state.evenly_distributed_target_state#,
    #target_state.outflow_target_state,
    #target_state.equal_prob_target_state,
]        

seeds = [0]

WEEK = 33
START_DAY = 0
START_HOUR = 6
PERIOD = timeInMinutes(hours=24)

###############################################################################

def lostTripsPlot(cities, policies, starv, cong):
    fig, subPlots = plt.subplots(nrows=1, ncols=len(cities), sharey=True)
    fig.suptitle("FOMO simulator - lost trips results", fontsize=15)
    w = 0.4
    if len(cities) == 1:
        subPlots = [ subPlots ]
    for city in range(len(cities)):
        subPlots[city].bar(policies, starv[city], w, label='Starvation')
        subPlots[city].bar(policies, cong[city], w, bottom=starv[city], label='Congestion')
        subPlots[city].set_xlabel(cities[city])
        if city == 0:
            subPlots[city].set_ylabel("Lost trips")
            subPlots[city].legend()

###############################################################################

if __name__ == "__main__":

    starvations = []
    congestions = []

    for city in cities:
        print("  city: ", city)

        starvations.append([])
        congestions.append([])

        for tstate in target_states:
            print("    target_state: ", tstate)

            initial_state = init_state.get_initial_state(source=init_state.cityBike, url=city,
                                                         week=WEEK, random_seed=0,
                                                         target_state=tstate)

            simulations = []

            for seed in seeds:
                print("      seed: ", seed)

                state_copy = copy.deepcopy(initial_state)
                state_copy.set_seed(seed)

                simul = sim.Simulator(
                    initial_state = state_copy,
                    policy = policies.GreedyPolicy(), 
                    start_time = timeInMinutes(days=START_DAY, hours=START_HOUR),
                    duration = PERIOD,
                    verbose = True,
                )
                
                simul.run()

                simulations.append(simul)

            metric = sim.Metric.merge_metrics([sim.metrics for sim in simulations])
            starvations[-1].append(metric.get_aggregate_value("lost_demand"))
            congestions[-1].append(metric.get_aggregate_value("congestion"))

    ###############################################################################

    city_names = [ url.split('/')[3].split('.')[0] for url in cities ]
    target_state_names = [ tstate.__name__.replace("_target_state", "") for tstate in target_states ]

    lostTripsPlot(city_names, target_state_names, starvations, congestions)
    plt.show()
