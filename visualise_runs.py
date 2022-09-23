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

        run = {}

        for line in lines:
            words = line.split(';')

            n = int(words[0])
            instance_name = words[1]
            analysis_name = words[2]
            trips = float(words[3])
            starvations = float(words[4])
            congestions = float(words[5])
            starvations_stdev = float(words[6])
            congestions_stdev = float(words[7])

            if instance_name not in run:
                run[instance_name] = {}
            run[instance_name][analysis_name] = (trips, starvations, congestions, starvations_stdev, congestions_stdev)

        instance_names = []

        starvations = []
        congestions = []

        starvations_stdev = []
        congestions_stdev = []

        for instance_name in run.keys():
            instance_names.append(instance_name)

            starvations.append([])
            congestions.append([])

            starvations_stdev.append([])
            congestions_stdev.append([])

            analysis_names = []

            for analysis_name in run[instance_name].keys():
                analysis_names.append(analysis_name)

                scale = 100 / run[instance_name][analysis_name][0]

                starvations[-1].append(scale * run[instance_name][analysis_name][1])
                congestions[-1].append(scale * run[instance_name][analysis_name][2])
                starvations_stdev[-1].append(scale * run[instance_name][analysis_name][3])
                congestions_stdev[-1].append(scale * run[instance_name][analysis_name][4])

        print(starvations)
        print(congestions)

        lostTripsPlot(instance_names, analysis_names, starvations, starvations_stdev, congestions, congestions_stdev)

        plt.show()
