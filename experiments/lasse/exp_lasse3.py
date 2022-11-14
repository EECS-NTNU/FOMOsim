#!/bin/python3
# exp_lasse3.py
# Contact Lasse Natvig if you have questions about this experiment

import copy
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike
import policies
import target_state
import demand
from helpers import *
import matplotlib.pyplot as plt
from output.plots import Surface3Dplot, Surface3DplotTripsProfit 

DURATION = timeInMinutes(hours=24)
INSTANCE_DIRECTORY="instances/extra"

instance = "OS_W33" # just one in this case

analyses = [
    dict(name="outflow-8",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
         numvehicles=8,
         day=0,
         hour=6), 
    dict(name="outflow-1",    #deviation_from_target_state
         target_state="OutflowTargetState",
         policy="GreedyPolicy",
         policyargs={'crit_weights':[0,0,0,1]},
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
    dict(name="equalprob",
         target_state="EqualProbTargetState",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6),
    dict(name="random",
         policy="RandomActionPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6),
    dict(name="do_nothing",
         numvehicles=0,
         day=0,
         hour=6),
    dict(name="do_nothing", # duplicate to get a square with right color
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
    for i in range(15): 
        bikes.append(startVal + i*200)       

    resultsStarvation = []   
    resultsCongestion = []
    resultsTotal = []
    resultTrips = []

    tripsStore = []  
    congStore = []
    starvStore = []
    
    resultProfit = []
    incomeTrip = 20 
    costStarvation = 2
    costCongestion = 4    

    print("  instance: ", instance)

    starvations.append([])
    congestions.append([])

    for analysis in analyses:
        print("    analysis: ", analysis["name"])   

        resultRowS = [] 
        resultRowC = [] 
        resultRowT = []
        resultRowTrips = []

        tripsRow = []
        congRow = []
        starvRow = []

        resultProfitRow = [] 
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
            trips = metric.get_aggregate_value("trips")
            scale = 100 / trips
            num_starvations = metric.get_aggregate_value("starvation") 
            starv = scale * num_starvations
            num_congestion = metric.get_aggregate_value("congestion") 
            cong = scale * num_congestion
            tot = starv + cong
            resultRowS.append(starv) 
            resultRowC.append(cong) 
            resultRowT.append(tot)

            trips = trips - num_starvations
            tripsRow.append(int(trips))
            congRow.append(int(num_congestion))
            starvRow.append(int(num_starvations))

            resultRowTrips.append(trips/200)
            resultProfitRow.append((trips*incomeTrip - num_starvations*costStarvation - num_congestion*costCongestion)/180000*100) 

        tripsStore.append(tripsRow)    
        congStore.append(congRow)    
        starvStore.append(starvRow)    

        resultsStarvation.append(resultRowS)
        resultsCongestion.append(resultRowC)
        resultsTotal.append(resultRowT)
        resultTrips.append(resultRowTrips)
        resultProfit.append(resultProfitRow)

    resultFile = open("output/costModel/simulatedTrips.txt", "w")
    for b in bikes:
        resultFile.write(str(b) + " ")
    resultFile.write("\n") 
    for p in range(len(policyNames)):
        resultFile.write(policyNames[p] + "\n")
        for i in range(len(bikes)):
            resultFile.write(str(tripsStore[p][i]) + " ")
        resultFile.write("\n")
        for i in range(len(bikes)):
            resultFile.write(str(starvStore[p][i]) + " ")
        resultFile.write("\n")
        for i in range(len(bikes)):
            resultFile.write(str(congStore[p][i]) + " ")
        resultFile.write("\n")
    resultFile.write("-end-\n")
    resultFile.close()
    
    fig, ax = Surface3Dplot(bikes, policyNames, resultsStarvation, "Starvation (" + '%' + " of trips)")
    fig, ax = Surface3Dplot(bikes, policyNames, resultsCongestion, "Congestion (" + '%' + " of trips)")
    fig, ax = Surface3Dplot(bikes, policyNames, resultsTotal, "Starvation + congestion (" + '%' + " of trips)")
    fig, ax = Surface3Dplot(bikes, policyNames, resultTrips, "No-of-trips/200")
    fig, ax = Surface3Dplot(bikes, policyNames, resultProfit, "Profit in percent of 180 kNOK")
    fig, ax = Surface3DplotTripsProfit(bikes, policyNames, resultTrips, resultProfit, "Oslo week 33, No of trips and profit")
    
    plt.show()

    print("-fin-")