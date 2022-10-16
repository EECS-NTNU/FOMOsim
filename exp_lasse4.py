#!/bin/python3
# exp_lasse4.py
import copy
# import settings
import sim
import init_state
import init_state.cityBike
import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
# from progress.bar import Bar
# import output
import target_state
import matplotlib.pyplot as plt
from helpers import *
from output.plots import lostTripsPlot

DURATION = timeInMinutes(hours=24)
INSTANCE_DIRECTORY="instances"
instances = ["OS_W31", "OS_W21"]
# instances = ["EH_W22", "EH_W31"]
analyses = [

    dict(name="do_nothing-1000",
         numbikes= 1000, # testing
         numvehicles=0,
         day=0,
         hour=6),

    dict(name="do_nothing",
         numvehicles=0,
         day=0,
         hour=6),

    #flat strategy
    dict(name="evenly",
         target_state="evenly_distributed_target_state",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0.25,0.25,0.25,0.25]},
         numvehicles=1,
         day=0,
         hour=6),    

    #deviation_from_target_state
    dict(name="outflow",
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
seeds = list(range(3))


if __name__ == "__main__":
    starvations = []
    congestions = []
    starvations_stdev = []
    congestions_stdev = []

    for instance in instances:
        print("  instance: ", instance)

        starvations.append([])
        congestions.append([])

        starvations_stdev.append([])
        congestions_stdev.append([])

        for analysis in analyses:
            print("    analysis: ", analysis["name"])

            tstate = None
            if "target_state" in analysis:
                tstate = getattr(target_state, analysis["target_state"])

            initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + instance, target_state=tstate)

            if analysis["numvehicles"] > 0:
                policyargs = analysis["policyargs"]
                policy = getattr(policies, analysis["policy"])(**policyargs)
                initial_state.set_vehicles([policy]*analysis["numvehicles"])

            # TODO wish to be able to set num_bikes here in a similar manner as numvehicles

            simulations = []
            for seed in seeds:
                print("      seed: ", seed)
                state_copy = copy.deepcopy(initial_state)
                state_copy.set_seed(seed)

                simul = sim.Simulator(
                    initial_state = state_copy,
                    start_time = timeInMinutes(days=analysis["day"], hours=analysis["hour"]),
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
    print(starvations)
    print(congestions)
    instance_names = instances
    analysis_names = [ analysis["name"] for analysis in analyses ]
    lostTripsPlot(instance_names, analysis_names, starvations, starvations_stdev, congestions, congestions_stdev)
    plt.show()