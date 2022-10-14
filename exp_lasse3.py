#!/bin/python3
# exp_lasse3.py
import copy
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike
import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import target_state
from helpers import *
from output.plots import Surface3Dplot, Surface3DplotTripsProfit 

DURATION = timeInMinutes(hours=24)
instances = ["EH_W22"]
analyses = [

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

policyNames = []
for ana in analyses:
    policyNames.append(ana[0])
policyIndices = range(len(policyNames))

seeds = [0]
 

if __name__ == "__main__":
    starvations = []
    congestions = []

    # set up number_of_bikes-values
    bikes = []
    startVal = 200
    for i in range(30): 
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

    # print("  instance: ", instance[0])
    starvations.append([])
    congestions.append([])
    for analysis in analyses:
        print("    analysis: ", analysis[0])        
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
            tstate = target_state.equal_prob_target_state
            state = init_state.read_initial_state("instances/" + instance, tstate);
            policy = policies.GreedyPolicy()
            state.set_vehicles([policy]) 

            simulations =[]     
            for seed in seeds:
                print("      seed: ", seed)
                state_copy = copy.deepcopy(state)
                state_copy.set_seed(seed)
                state_copy.set_vehicles(analysis[3])
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


            trips = trips - num_starvations # todo, into variable for speed
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
    
    print(resultsStarvation)
    print(resultsCongestion)
    print(resultsTotal)
    print(resultTrips)
    print(resultProfit)

#    fig, ax = Surface3Dplot(bikes, policyNames, resultsStarvation, "Starvation (" + '%' + " of trips)")
#    fig, ax = Surface3Dplot(bikes, policyNames, resultsCongestion, "Congestion (" + '%' + " of trips)")
    fig, ax = Surface3Dplot(bikes, policyNames, resultsTotal, "Starvation + congestion (" + '%' + " of trips)")
    fig, ax = Surface3Dplot(bikes, policyNames, resultTrips, "No-of-trips/200")
    fig, ax = Surface3Dplot(bikes, policyNames, resultProfit, "Profit in percent of 180 kNOK")

    fig, ax = Surface3DplotTripsProfit(bikes, policyNames, resultTrips, resultProfit, "Oslo week 33, No of trips and profit")
    
    plt.show()
    print(" bye bye")
