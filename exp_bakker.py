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
import policies.gleditsch_hagen

from progress.bar import Bar

import output
import target_state
import matplotlib.pyplot as plt

from helpers import *

###############################################################################
#                               FOMO SIMULATOR                                #
###############################################################################

DURATION = timeInMinutes(hours=2)

NUM_SEEDS = 1

# Enter instance definition here.  For numbikes and numstations, enter 'None' to use dataset default
instances = [
    # Name,         URL,                                                          numbikes, numstations, week, day, hour
#    ("Oslo",        "https://data.urbansharing.com/oslobysykkel.no/trips/v1/",        None,        None,   33,   0,    6 ),
#    ("Bergen",      "https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",      None,        None,   33,   0,    6 ),
    ("Trondheim",   "https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",   None,        None,   33,   0,    8 ),
#   ("Oslo-vinter", "https://data.urbansharing.com/oslovintersykkel.no/trips/v1/",    None,        None,   33,   0,    6 ),
#   ("Edinburgh",   "https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/", None,        None,   33,   0,    6 ),
]

if True:
    analyses = [
        # Name,             target_state,                                   policy                                              numvehicles
        ("grd_crt_even",    target_state.outflow_target_state,   policies.GreedyPolicy(crit_weights=[0,0,0,1]),            1),
        ("gh_even",    target_state.outflow_target_state,   policies.gleditsch_hagen.GleditschHagenPolicy(),            1),
    ]    
elif False:
    # Enter analysis definition here
    analyses = [
        # Name,             target_state,                                   policy                                              numvehicles
        # ("do_nothing",      target_state.outflow_target_state,              policies.DoNothing(),                               1),
        # ("grd_trgt_outfl",  target_state.outflow_target_state,              policies.GreedyPolicy(
        #                                                                     crit_weights=[0,0,0,1]), 1),
        # ("grd_crt_outfl",   target_state.outflow_target_state,              policies.GreedyPolicy(crit_weights=[0.25,0.25,0.25,0.25]),            1),
        ("grd_crt_even",    target_state.evenly_distributed_target_state,   policies.GreedyPolicy(crit_weights=[0.25,0.25,0.25,0.25]),            1),
        # ("grd_crt_equal",   target_state.equal_prob_target_state,           policies.GreedyPolicy(crit_weights=[0.25,0.25,0.25,0.25]),            1),
        #("grd_old",     target_state.outflow_target_state,       policies.GreedyPolicyOld(),                           1),
    ]      
    #outflow_target_state, evenly_distributed_target_state, equal_prob_target_state
else:
    analyses1 = [
        # Name,             target_state,                       policy                                      numvehicles
        
        ("1a",  target_state.evenly_distributed_target_state,   policies.GreedyPolicy(crit_weights=[0,0,0,1]),             1),
        ("1b",  target_state.outflow_target_state,              policies.GreedyPolicy(crit_weights=[0,0,0,1]),             1),
        ("1c",  target_state.equal_prob_target_state,           policies.GreedyPolicy(crit_weights=[0,0,0,1]),             1),
    ]      
    
    analyses2 = [
        # Name,             target_state,                       policy                              numvehicles
        ("2a",  target_state.outflow_target_state,   policies.DoNothing(),                          1),
        ("2b",  target_state.outflow_target_state,   policies.GreedyPolicy(crit_weights=[0,0,0,1]),                1),
        ("2c",  target_state.outflow_target_state,   policies.GreedyPolicy(crit_weights=[0.25,0.25,0.25,0.25]),                           1),
    ]  
    
    analyses3 = [
        # Name,             target_state,                       policy                              numvehicles
        ("3_1",  target_state.outflow_target_state,   policies.GreedyPolicy(crit_weights=[0,0,0,1]),                           1),
        ("3_2",  target_state.outflow_target_state,   policies.GreedyPolicy(crit_weights=[0,0,0,1]),                           2),
        ("3_3",  target_state.outflow_target_state,   policies.GreedyPolicy(crit_weights=[0,0,0,1]),                           3),
    ]  
 
analyses = analyses

seeds = list(range(NUM_SEEDS))  # [0,1,2,....,N-1]
#seeds = [0]


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
            starvations[-1].append(100 * metric.get_aggregate_value("starvation") / metric.get_aggregate_value("trips"))
            congestions[-1].append(100 * metric.get_aggregate_value("congestion") / metric.get_aggregate_value("trips"))

    ###############################################################################

    city_names = [ instance[0] for instance in instances ]
    analysis_names = [ analysis[0] for analysis in analyses ]

    lostTripsPlot(city_names, analysis_names, starvations, congestions)
    #plt.savefig('violations_plot.png')
    plt.show()
