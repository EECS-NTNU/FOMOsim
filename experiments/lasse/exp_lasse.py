#!/bin/python3
# exp_lasse.py
# Contact Lasse Natvig if you have questions about this experiment
import copy
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike
import policies
import target_state
import demand
from output.plots import lostTripsPlot
from helpers import *

DURATION = timeInMinutes(hours=24)
INSTANCE_DIRECTORY="instances"
instances = ["OS_W31"]
analyses = [
    dict(name="do_nothing",
         numvehicles=0,
         day=0,
         hour=6),
    dict(name="evenly",     #flat strategy
         target_state="EvenlyDistributedTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0.25,0.25,0.25,0.25]},
         numvehicles=1,
         day=0,
         hour=6),    
    dict(name="outflow",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=1,
         day=0,
         hour=6),     
    dict(name="equalprob",
         target_state="EqualProbTargetState",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6),
]    

seeds = list(range(2))

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
                tstate = getattr(target_state, analysis["target_state"])()

            initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + instance,
                                                          number_of_stations=analysis.get("numstations", None),
                                                          number_of_bikes=analysis.get("numbikes", None),
                                                          )
            
            if analysis["numvehicles"] > 0:
                policyargs = analysis["policyargs"]
                policy = getattr(policies, analysis["policy"])(**policyargs)
                initial_state.set_sb_vehicles([policy]*analysis["numvehicles"])

            simulations = []
            for seed in seeds:
                print("      seed: ", seed)
                state_copy = copy.deepcopy(initial_state)
                state_copy.set_seed(seed)
    
                simul = sim.Simulator(
                    initial_state = state_copy,
                    target_state = tstate,
                    demand = demand.Demand(),
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

    instance_names = instances
    analysis_names = [ analysis["name"] for analysis in analyses ]

    lostTripsPlot(instance_names, analysis_names, starvations, starvations_stdev, congestions, congestions_stdev)
