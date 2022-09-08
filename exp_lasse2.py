#!/bin/python3
import copy
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike
import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import target_state
import matplotlib.pyplot as plt
from matplotlib import cm 
import numpy as np
from helpers import *

def Surface3Dplot(bikes, policyNames, values, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    policiesPosition = range(len(policyNames))
    bikes, policiesPosition = np.meshgrid(bikes, policiesPosition)
    values2D = bikes + policiesPosition*100 # 2D numpyarray
    Z = values2D # Z behaves as a pointer/reference to values2D

    # copy results from 2D list to numpyarray
    for p in range(len(policyNames)):
        for b in range(len(bikes[p])):
            # print("p:b", p, ":", b)
            Z[p][b] = values[p][b] 

    surf = ax.plot_surface(bikes, policiesPosition, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_xlabel("number of bikes")
    ax.set_ylabel("rebalancing policy")
    ax.set_yticks(range(len(policyNames)))
    ax.set_yticklabels(policyNames)
    ax.set_zlim([0, 150])
    # https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    fig.colorbar(surf, shrink=0.3, aspect=5, )
    fig.suptitle(title)
    return fig, ax

DURATION = timeInMinutes(hours=48)
instances = [ ("Oslo", "https://data.urbansharing.com/oslobysykkel.no/trips/v1/", None,  None,   33,   0,    DURATION ),
]
analyses = [
    # Name,        target_state,                                 policy,                  numvehicles
    ("equalprob-2",  target_state.equal_prob_target_state,         policies.GreedyPolicy(),           2),
    ("equalprob-1",  target_state.equal_prob_target_state,         policies.GreedyPolicy(),           1),
    ("outflow-2",    target_state.outflow_target_state,            policies.GreedyPolicy(),           2),
    ("outflow-1",    target_state.outflow_target_state,            policies.GreedyPolicy(),           1),
    ("evenly-2",     target_state.evenly_distributed_target_state, policies.GreedyPolicy(),           2),
    ("evenly-1",     target_state.evenly_distributed_target_state, policies.GreedyPolicy(),           1),
    ("random-2",     target_state.evenly_distributed_target_state, policies.RandomActionPolicy(),     2),
    ("random-1",     target_state.evenly_distributed_target_state, policies.RandomActionPolicy(),     1),
    ("do_nothing",   target_state.evenly_distributed_target_state, policies.DoNothing(),              1),
]

policyNames = []
for ana in analyses:
    policyNames.append(ana[0])
policyIndices = range(len(policyNames))

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
seeds = [0,1,2]

def lostTripsPlot(cities, policies, starv, starv_stdev, cong, cong_stdev):
    fig, subPlots = plt.subplots(nrows=1, ncols=len(cities), sharey=True)
    fig.suptitle("FOMO simulator - lost trips results", fontsize=15)
    
    if len(cities) == 1:
        subPlots = [ subPlots ]
    w = 0.5
    pos = []
    for city in range(len(cities)):
        pos.append([])
        for i in range(len(cong[city])):
            pos[city].append(starv[city][i] + cong[city][i])

        subPlots[city].bar(policies, starv[city], w, label='Starvation')
        subPlots[city].errorbar(policies, starv[city], yerr = starv_stdev[city], fmt='none', ecolor='red')
        subPlots[city].bar(policies, cong[city], w, bottom=starv[city], label='Congestion')
        
        # skew the upper error-bar with delta to avoid that they can overwrite each other
        delta = 0.05
        policiesPlussDelta = []
        for i in range(len(policies)):
            policiesPlussDelta.append(i + delta) 
        subPlots[city].errorbar(policiesPlussDelta, pos[city], yerr= cong_stdev[city], fmt='none', ecolor='black')
        subPlots[city].set_xlabel(cities[city])
        if city == 0:
            subPlots[city].set_ylabel("Violations (% of total number of trips)")
            subPlots[city].legend()

if __name__ == "__main__":
    starvations = []
    congestions = []

    # set up number_of_bikes-values
    bikes = []
    startVal = 1000
    for i in range(5): # 12
        bikes.append(startVal + i*600) 

    resultsStarvation = []  
    resultsCongestion = []
    resultsTotal = []    

    for instance in instances:
        print("  instance: ", instance[0])
        starvations.append([])
        congestions.append([])
        for analysis in analyses:
            print("    analysis: ", analysis[0])        
            resultRowS = [] 
            resultRowC = [] 
            resultRowT = [] 
            for b in bikes:
                print( "   number of bikes: ", b)

                if instance[0] == "Oslo":
                    initial_state = init_state.get_initial_state(source=init_state.cityBike, url=instance[1], week=instance[4],
                                                            fromInclude=[2020, 7], toInclude= [2022,8],
                                                            random_seed=0, number_of_stations=instance[3], number_of_bikes=b,
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

    plt.show()
    print(" bye bye")
