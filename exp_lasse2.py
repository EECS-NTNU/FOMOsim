#!/bin/python3
import copy
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike
import policies
import target_state
import demand
import matplotlib.pyplot as plt
from helpers import *
from output.plots import Surface3Dplot, Surface3DplotFraction 
      
DURATION = timeInMinutes(hours=24)
INSTANCE_DIRECTORY="instances/extra"

instances = ["OS_W33"] # just one in this case

analyses = [
    dict(name="outflow-16",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=16,
         day=0,
         hour=6), 
    dict(name="outflow-12",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=12,
         day=0,
         hour=6), 
    dict(name="outflow-10",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=10,
         day=0,
         hour=6), 
    dict(name="outflow-8",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=8,
         day=0,
         hour=6), 
    dict(name="outflow-7",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=7,
         day=0,
         hour=6), 
    dict(name="outflow-6",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=6,
         day=0,
         hour=6), 
    dict(name="outflow-5",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=5,
         day=0,
         hour=6), 
    dict(name="outflow-4",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=4,
         day=0,
         hour=6), 
    dict(name="outflow-3",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=3,
         day=0,
         hour=6), 
    dict(name="outflow-2",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=2,
         day=0,
         hour=6), 
    dict(name="outflow-1",    #deviation_from_target_state
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
    dict(name="evenly",     #flat strategy
         target_state="EvenlyDistributedTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0.25,0.25,0.25,0.25]},
         numvehicles=1,
         day=0,
         hour=6),    
    
    dict(name="do_nothing",
         numvehicles=0,
         day=0,
         hour=6),
    dict(name="do_nothing",
         numvehicles=0,
         day=0,
         hour=6),
]    
 
policyNames = []
for ana in analyses:
    policyNames.append(ana["name"])
policyIndices = range(len(policyNames))

seeds = list(range(10))

if __name__ == "__main__":
    starvations = []
    congestions = []

    bikes = []
    startVal = 1600
    for i in range(14): 
        bikes.append(startVal + i*200)      

    resultsStarvation = []  
    resultsCongestion = []
    resultsTotal = []    

    for instance in instances:
        print("  instance: ", instance)

    
        starvations.append([])
        congestions.append([])

        for analysis in analyses:
            print("    analysis: ", analysis["name"])       
            resultRowS = [] 
            resultRowC = [] 
            resultRowT = [] 

            for b in bikes:
                print( "   number of bikes: ", b)

                tstate = None
                if "target_state" in analysis:
                    tstate = getattr(target_state, analysis["target_state"])()

                initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + instance,  
                    number_of_stations=analysis.get("numstations", None), number_of_bikes=b)
                    
                if analysis["numvehicles"] > 0:
                    policyargs = analysis["policyargs"]
                    policy = getattr(policies, analysis["policy"])(**policyargs)
                    initial_state.set_vehicles([policy]*analysis["numvehicles"])

                simulations =[]     
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
                starv = scale * metric.get_aggregate_value("starvation")
                cong = scale * metric.get_aggregate_value("congestion")
                tot = starv + cong
                resultRowS.append(starv) 
                resultRowC.append(cong) 
                resultRowT.append(tot) 

            resultsStarvation.append(resultRowS)
            resultsCongestion.append(resultRowC)
            resultsTotal.append(resultRowT)

    print(resultsStarvation)
    print(resultsCongestion)
    print(resultsTotal)

    fig, ax = Surface3Dplot(bikes, policyNames, resultsStarvation, "Starvation (" + '%' + " of trips)")
    fig, ax = Surface3Dplot(bikes, policyNames, resultsCongestion, "Congestion (" + '%' + " of trips)")
    fig, ax = Surface3Dplot(bikes, policyNames, resultsTotal, "Starvation + congestion (" + '%' + " of trips)")
    fig, ax = Surface3DplotFraction(bikes, policyNames, resultsStarvation, resultsCongestion, "Oslo week 33, Lost trips (" + '%' + ") and congestion-starvation ratio")
    
    plt.show()

    print(" - fin - ")
