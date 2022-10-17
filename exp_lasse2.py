#!/bin/python3
import copy
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike
import policies
import target_state
import matplotlib.pyplot as plt
from helpers import *
from output.plots import Surface3Dplot, Surface3DplotFraction 
      
DURATION = timeInMinutes(hours=24)
INSTANCE_DIRECTORY="instances"

instances = ["OS_W31"] # just one in this case

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
 
policyNames = []
for ana in analyses:
    policyNames.append(ana["name"])
policyIndices = range(len(policyNames))

seeds = list(range(2))

if __name__ == "__main__":
    starvations = []
    congestions = []

    bikes = []
    startVal = 1000
    for i in range(3): 
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
                    tstate = getattr(target_state, analysis["target_state"])

                initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + instance, target_state=tstate)
                
                if analysis["numvehicles"] > 0:
                    policyargs = analysis["policyargs"]
                    policy = getattr(policies, analysis["policy"])(**policyargs)
                    initial_state.set_vehicles([policy]*analysis["numvehicles"])

                # TODO prøv sette antall sykler her ... // AVENTER diskusjon mandag  --- sendte mail Steffen 

                simulations =[]     
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

    pass
