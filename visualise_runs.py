#!/bin/python3
"""
FOMO simulator main program
"""
import copy
import os
import shutil
import json

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

###############################################################################

def lostTripsPlot(cities, policies, starv, starv_stdev, cong, cong_stdev):
    fig, subPlots = plt.subplots(nrows=1, ncols=len(cities), sharey=True)
    fig.suptitle("FOMO simulator - lost trips results", fontsize=15)
    
    if len(cities) == 1:
        subPlots = [ subPlots ]
    w = 0.3
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


###############################################################################

if __name__ == "__main__":

    with open("output.csv", "r") as infile:
        lines = infile.readlines()

        instance_names = []
        analysis_names = []

        starvations = []
        congestions = []

        starvations_stdev = []
        congestions_stdev = []

        for line in lines:
            words = line.split(';')

            n = int(words[0])
            instance_name = words[1]
            analysis_name = words[2]
            trips = float(words[3])
            starvation = float(words[4])
            congestion = float(words[5])
            starvation_stdev = float(words[6])
            congestion_stdev = float(words[7])

            instance_names.append(instance_name)
            analysis_names.append(analysis_name)
            starvations.append(starvation)
            congestions.append(congestions)
            starvations_stdev.append(starvation_stdev)
            congestions_stdev.append(congestion_stdev)

        lostTripsPlot(instance_names, analysis_names, starvations, starvations_stdev, congestions, congestions_stdev)

        plt.show()
