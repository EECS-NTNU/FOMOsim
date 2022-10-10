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
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
from helpers import *

def Surface3Dplot(bikes, policyNames, values, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    policiesPosition = range(len(policyNames))
    bikes, policiesPosition = np.meshgrid(bikes, policiesPosition)
    values2D = bikes + policiesPosition*100 # 2D numpyarray
    Z = values2D # Z behaves as a pointer/reference to values2D
    for p in range(len(policyNames)):  # copy results from 2D list to numpyarray
        for b in range(len(bikes[p])):
            Z[p][b] = values[p][b] 
    ax.set_xlabel("number of bikes")
    ax.set_ylabel("rebalancing policy")
    ax.set_yticks(range(len(policyNames)))
    ax.set_yticklabels(policyNames)
    norm = colors.Normalize(vmin=0, vmax=100)
    scmap = plt.cm.ScalarMappable(norm = norm, cmap="coolwarm")
    surf = ax.plot_surface(bikes, policiesPosition, Z, facecolors=scmap.to_rgba(Z), linewidth=0, antialiased=False, shade = False)
    ax.set_zlim([0, 100])
    # https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    fig.colorbar(scmap, shrink=0.3, aspect=5, )
    fig.suptitle(title)
    return fig, ax

def Surface3DplotFraction(bikes, policyNames, starv, cong, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    policiesPosition = range(len(policyNames))
    bikes, policiesPosition = np.meshgrid(bikes, policiesPosition)
    Z = bikes + policiesPosition # 2D numpyarray, addition is not used, but gives Z the right shape, Z behaves now as a reference to a 2D-matrix
    Rint = np.copy(Z)
    R = Rint.astype(float) 
    for p in range(len(policyNames)): # copy results from 2D list to numpyarray
        for b in range(len(bikes[p])):
            Z[p][b] = starv[p][b] + cong[p][b] # Z-value to plot, with zero at zero, in range 0 - 120 or more
    for p in range(len(policyNames)): # copy results from 2D list to numpyarray
        for b in range(len(bikes[p])):
            R[p][b] = (starv[p][b]/Z[p][b])*100 # Starvation in % of Lost trips  
    norm = colors.Normalize(vmin=0, vmax=100)
    norm2 = colors.Normalize(vmin=0.0, vmax=100)
    scmap = plt.cm.ScalarMappable(norm = norm, cmap="coolwarm")
    scmap2 = plt.cm.ScalarMappable(norm = norm2, cmap=cm.BrBG)
    surf = ax.plot_surface(bikes, policiesPosition, Z, facecolors=scmap.to_rgba(Z), shade=False)
    surf2 = ax.plot_surface(bikes, policiesPosition, (R*0)-100, facecolors=scmap2.to_rgba(R), shade=False)
    ax.set_zlim(-100, 100) 
    ax.set_xlabel("number of bikes", weight='bold')
    ax.set_ylabel("rebalancing policy",  weight='bold')
    ax.set_yticks(range(len(policyNames)))
    ax.set_yticklabels(policyNames)
    # https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    # yaxis-set_(policies, policyNames)
    cb = fig.colorbar(scmap, shrink=0.2, aspect=5, location = "right")
    cb2 = fig.colorbar(scmap2, shrink=0.2, aspect=5, location = "bottom")
    cb.set_label(label="Lost trips in " + '%' + " of total trips", size = 9)
    cb2.set_label(label = "Congestion - starvation ratio", size = 9)
    fig.suptitle(title)
    return fig, ax

def Surface3DplotTripsProfit(bikes, policyNames, trips, profit, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    policiesPosition = range(len(policyNames))
    bikes, policiesPosition = np.meshgrid(bikes, policiesPosition)
    Z = bikes + policiesPosition # 2D numpyarray, addition is not used, but gives Z the right shape, Z behaves now as a reference to a 2D-matrix
    Rint = np.copy(Z)
    R = Rint.astype(float) 
    for p in range(len(policyNames)): # copy results from 2D list to numpyarray
        for b in range(len(bikes[p])):
            Z[p][b] = trips[p][b]  # Z-value to plot, with zero at zero, in range 0 - 120 or more
    for p in range(len(policyNames)):
        for b in range(len(bikes[p])):
            R[p][b] = profit[p][b] 
    norm = colors.Normalize(vmin=0, vmax=100)
    norm2 = colors.Normalize(vmin=0.0, vmax=100)
    scmap = plt.cm.ScalarMappable(norm = norm, cmap="coolwarm")
    scmap2 = plt.cm.ScalarMappable(norm = norm2, cmap=cm.BrBG)
    surf = ax.plot_surface(bikes, policiesPosition, Z, facecolors=scmap.to_rgba(Z), shade=False) # upper surface
    surf2 = ax.plot_surface(bikes, policiesPosition, (R*0)-100, facecolors=scmap2.to_rgba(R), shade=False) # lower flat surface
    ax.set_zlim(-100, 100) 
    ax.set_xlabel("number of bikes", weight='bold')
    ax.set_ylabel("rebalancing policy",  weight='bold')
    ax.set_yticks(range(len(policyNames)))
    ax.set_yticklabels(policyNames)
    # https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    # yaxis-set_(policies, policyNames)
    cb = fig.colorbar(scmap, shrink=0.2, aspect=5, location = "right")
    cb2 = fig.colorbar(scmap2, shrink=0.2, aspect=5, location = "bottom")
    cb.set_label(label="Trips/200", size = 9)
    cb2.set_label(label = "Profit/180kNOK", size = 9)
    fig.suptitle(title)
    return fig, ax

DURATION = timeInMinutes(hours=24)
instances = [ ("Oslo", "https://data.urbansharing.com/oslobysykkel.no/trips/v1/", None,  None,   33,   0,    DURATION )]
analyses = [
#    Name,        target_state,                                 policy,                  numvehicles
    # ("equalprob-2",  target_state.equal_prob_target_state,         policies.GreedyPolicy(),           2),
    # ("equalprob-1",  target_state.equal_prob_target_state,         policies.GreedyPolicy(),           1),
    # ("outflow-8",    target_state.outflow_target_state,            policies.GreedyPolicy(),           8), # TODO, fix this UGLY copy and paste code
    # ("outflow-7",    target_state.outflow_target_state,            policies.GreedyPolicy(),           7),
    # ("outflow-6",    target_state.outflow_target_state,            policies.GreedyPolicy(),           6),
    # # ("outflow-5",    target_state.outflow_target_state,            policies.GreedyPolicy(),           5),
    # ("outflow-4",    target_state.outflow_target_state,            policies.GreedyPolicy(),           4),
    # ("outflow-3",    target_state.outflow_target_state,            policies.GreedyPolicy(),           3),
    # ("outflow-2",    target_state.outflow_target_state,            policies.GreedyPolicy(),           2),
    # ("outflow-1",    target_state.outflow_target_state,            policies.GreedyPolicy(),           1),
    # # ("evenly-2",     target_state.evenly_distributed_target_state, policies.GreedyPolicy(),           2),
    # ("evenly-1",     target_state.evenly_distributed_target_state, policies.GreedyPolicy(),           1),
    # ("random-2",     target_state.evenly_distributed_target_state, policies.RandomActionPolicy(),     2),
    ("random-1",     target_state.evenly_distributed_target_state, policies.RandomActionPolicy(),     1),
    ("do_nothing",   target_state.evenly_distributed_target_state, policies.DoNothing(),              1),
]

policyNames = []
for ana in analyses:
    policyNames.append(ana[0])
policyIndices = range(len(policyNames))

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
seeds = [0 ] 

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
    startVal = 2000
    for i in range(5): 
        bikes.append(startVal + i*200)       

    resultsStarvation = []  
    resultsCongestion = []
    resultsTotal = []
    resultTrips = []
    tripsStore =[]  
    resultProfit = []
    incomeTrip = 20 
    costStarvation = 2
    costCongestion = 4    

    for instance in instances:
        print("  instance: ", instance[0])
        starvations.append([])
        congestions.append([])
        for analysis in analyses:
            print("    analysis: ", analysis[0])        
            resultRowS = [] 
            resultRowC = [] 
            resultRowT = []
            resultRowTrips = []
            tripsRow = []
            resultProfitRow = [] 
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
                tripsRow.append(trips)
                resultRowTrips.append(trips/200)
                resultProfitRow.append((trips*incomeTrip - num_starvations*costStarvation - num_congestion*costCongestion)/180000*100) 
            tripsStore.append(tripsRow)    
            resultsStarvation.append(resultRowS)
            resultsCongestion.append(resultRowC)
            resultsTotal.append(resultRowT)
            resultTrips.append(resultRowTrips)
            resultProfit.append(resultProfitRow)

    print("number-of-Bikes-values")
    for b in bikes:
        print(b, end="")
    print()

    for p in range(len(policyNames)):
        print(policyNames[p], " ")
        for i in range(len(bikes)):
            print(tripsStore[p][i], " ", end="")
        print()
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
