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
from output.plots import lostTripsPlot

###############################################################################
# Duration of each simulation run
DURATION = timeInMinutes(hours=24)

# Enter instance definition here.  For numbikes and numstations, enter 'None' to use dataset default
instances = [
    # Name,         URL,                                                          numbikes, numstations, week, day, hour
    ("Bergen",      "https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",      None,        None,   25,   0,    6 ),
]

# Enter analysis definition here
analyses = [
    # Name,        target_state,                                  policy,                   numvehicles     trafficMultiplier 
    ("do_no",   target_state.evenly_distributed_target_state,     policies.DoNothing(),          1,     1.0),
    ("out-1",   target_state.outflow_target_state,                policies.GreedyPolicy(),       1,     1.0),
    ("out-2",   target_state.outflow_target_state,                policies.GreedyPolicy(),       2,     1.0),
    ("do_no-1.5",   target_state.evenly_distributed_target_state, policies.DoNothing(),          1,     1.5),
    ("out-1-1.5",   target_state.outflow_target_state,            policies.GreedyPolicy(),       1,     1.5),
    ("out-2-1.5",   target_state.outflow_target_state,            policies.GreedyPolicy(),       2,     1.5),
    ("do_no-2.0",   target_state.evenly_distributed_target_state, policies.DoNothing(),          1,     2.0),
    ("out-1-2.0",   target_state.outflow_target_state,            policies.GreedyPolicy(),       1,     2.0),
    ("out-2-2.0",   target_state.outflow_target_state,            policies.GreedyPolicy(),       2,     2.0),
    ("do_no-4.0",   target_state.evenly_distributed_target_state, policies.DoNothing(),          1,     4.0),
    ("out-1-4.0",   target_state.outflow_target_state,            policies.GreedyPolicy(),       1,     4.0),
    ("out-2-4.0",   target_state.outflow_target_state,            policies.GreedyPolicy(),       2,     4.0),
]        
seeds = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
seeds = [ 0, 1] 
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
                                                  
            if instance[0] == "Bergen":
                initial_state = init_state.get_initial_state(source=init_state.cityBike, url=instance[1], week=instance[4],
                                                        fromInclude=[2018, 9], toInclude= [2021, 9],
                                                        random_seed=0, number_of_stations=instance[3], number_of_bikes=instance[2],
                                                        target_state=analysis[1], trafficMultiplier=analysis[4])        
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
