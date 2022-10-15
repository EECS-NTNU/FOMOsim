#!/bin/python3
# exp_lasse.py
import copy
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike
from output.plots import lostTripsPlot
from helpers import *

DURATION = timeInMinutes(hours=24)
instances = ["EH_W22"]
analyses = [
    dict(name="do_nothing",
         numvehicles=0,
         day=0,
         hour=6),
    dict(name="evenly",     #flat strategy
         target_state="evenly_distributed_target_state",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0.25,0.25,0.25,0.25]},
         numvehicles=1,
         day=0,
         hour=6),    
    dict(name="outflow",    #deviation_from_target_state
         target_state="outflow_target_state",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=1,
         day=0,
         hour=6),     
    dict(name="equalprob",
         target_state="equal_prob_target_state",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6),
]    

seeds = list(range(5))

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
