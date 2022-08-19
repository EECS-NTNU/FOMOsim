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
    ("Oslo",        "https://data.urbansharing.com/oslobysykkel.no/trips/v1/",        None,        None,   33,   0,    6 ),
    ("Bergen",      "https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",      None,        None,   33,   0,    6 ),
    ("Trondheim",   "https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",   None,        None,   33,   0,    6 ),
#   ("Oslo-vinter", "https://data.urbansharing.com/oslovintersykkel.no/trips/v1/",    None,        None,   33,   0,    6 ),
#   ("Edinburgh",   "https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/", None,        None,   33,   0,    6 ),
]

# Enter analysis definition here
analyses = [
    # Name,        target_state,                                 policy,                  numvehicles
    ("do_nothing", target_state.evenly_distributed_target_state, policies.DoNothing(),              1),
    ("evenly",     target_state.evenly_distributed_target_state, policies.GreedyPolicy(),           1),
    ("outflow",    target_state.outflow_target_state,            policies.GreedyPolicy(),           1),
    ("equalprob",  target_state.equal_prob_target_state,         policies.GreedyPolicy(),           1),
]        

seeds = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

###############################################################################

def lostTripsPlot(cities, policies, starv, cong):
    fig, subPlots = plt.subplots(nrows=1, ncols=len(cities), sharey=True)
    fig.suptitle("FOMO simulator", fontsize=15)
    w = 0.4
    if len(cities) == 1:
        subPlots = [ subPlots ]
    for city in range(len(cities)):
        subPlots[city].bar(policies, starv[city], w, label='Starvation')
        subPlots[city].bar(policies, cong[city], w, bottom=starv[city], label='Congestion')
        subPlots[city].set_xlabel(cities[city])
        if city == 0:
            subPlots[city].set_ylabel("Violations (% of total number of trips)")
            subPlots[city].legend()

###############################################################################

if __name__ == "__main__":

    starvations = []
    congestions = []

    for instance in instances:
        print("  city: ", instance[0])

        starvations.append([])
        congestions.append([])

        for analysis in analyses:
            print("    analysis: ", analysis[0])

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
            starvations[-1].append(100 * metric.get_aggregate_value("lost_demand") / metric.get_aggregate_value("trips"))
            congestions[-1].append(100 * metric.get_aggregate_value("congestion") / metric.get_aggregate_value("trips"))

    ###############################################################################

    city_names = [ instance[0] for instance in instances ]
    analysis_names = [ analysis[0] for analysis in analyses ]

    lostTripsPlot(city_names, analysis_names, starvations, congestions)
    plt.show()
