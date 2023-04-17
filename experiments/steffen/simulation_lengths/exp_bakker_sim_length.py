#!/bin/python3
"""
FOMO simulator main program
"""

####################################
# set the right working directory #
###################################

import os 
import sys
from pathlib import Path
from turtle import color

path = Path(__file__).parents[3]
os.chdir(path)
#print(os. getcwd())

sys.path.insert(0, '') #make sure the modules are found in the new working directory

##############################################################

import copy

import settings
import sim
import init_state
import init_state.fosen_haldorsen
import init_state.cityBike
import itertools
import pandas as pd
from functools import reduce

import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import policies.gleditsch_hagen
import demand
from progress.bar import Bar

import output
import target_state
import matplotlib.pyplot as plt

from helpers import *

import matplotlib.pyplot as plt
import copy
import matplotlib.dates as mdates
from matplotlib import gridspec

###############################################################################
#                               FOMO SIMULATOR                                #
###############################################################################

START_TIME = timeInMinutes(hours=7)
NUM_WEEKS = 4
NUM_DAYS = NUM_WEEKS*7
DURATION = timeInMinutes(hours=24*NUM_DAYS)
WEEK = 34


def main(instance):

    ###############################################################################
    # Get initial state

    
    
    state = init_state.read_initial_state("instances/"+instance);
    state.set_seed(1)

    ###############################################################################
    # Set up policy

    tstate = target_state.USTargetState()

    #policy = policies.GreedyPolicy()

    state.set_vehicles([])  #this is DoNothing policy

    ###############################################################################
    # Set up demand

    dmand = demand.Demand()

    ###############################################################################
    # Set up simulator

    simulator = sim.Simulator(
        initial_state = state,
        target_state = tstate,
        demand = dmand,
        start_time = START_TIME,
        duration = DURATION,
        verbose = True,
    )
    simulator.run()

    ###############################################################################
    # Output
    
    print(f"Simulation time = {DURATION} minutes")
    print(f"Total requested trips = {simulator.metrics.get_aggregate_value('trips')}")
    print(f"Starvations = {simulator.metrics.get_aggregate_value('starvation')}")
    print(f"Congestions = {simulator.metrics.get_aggregate_value('congestion')}")

    #output.write_csv(simulator, "output.csv", WEEK, hourly = False)
    #output.visualize_trips([simulator], title=("Week " + str(WEEK)), week=WEEK)
    #output.visualize_starvation([simulator], title=("Week " + str(WEEK)), week=WEEK)
    #output.visualize_congestion([simulator], title=("Week " + str(WEEK)), week=WEEK)

    metric = simulator.metrics
    def plot_lost_trips_over_time(metric,instance):
        
        timeline =  [i[0] for i in metric.metrics['trips']] + [i[0] for i in metric.metrics['congestion']] + [i[0] for i in metric.metrics['starvation']]

        trips_df = pd.DataFrame(metric.metrics['trips'], columns =['time', 'trips'])
        congestions_df = pd.DataFrame(metric.metrics['congestion'], columns =['time', 'congestions'])
        starvations_df = pd.DataFrame(metric.metrics['starvation'], columns =['time', 'starvations'])   
        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['time'],
                                                    how='outer'), [trips_df,congestions_df,starvations_df]) #.fillna('void')
        df_merged = df_merged.sort_values('time')
        df_merged = df_merged.interpolate(method='linear', limit_direction=None, axis=0)
        df_merged['cong'] = df_merged['congestions']/df_merged['trips']*100
        df_merged['starv'] = df_merged['starvations']/df_merged['trips']*100
        df_merged['time'] = df_merged['time']/(60*24)
        x = df_merged['time']
        df_merged = df_merged.set_index('time')
        
        y = np.vstack([df_merged['cong'], df_merged['starv']])
        labels = ["congestions ", "starvations"]

        fig, ax = plt.subplots()
        color_map = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]
        ax.stackplot(x, df_merged['cong'], df_merged['starv'],labels=labels,colors = color_map)
        ax.legend(loc='upper left')
        ax.set_xlabel("time in days")
        ax.set_ylabel("lost trips (%)")
        ax.set_title(instance)
        
        location = 'C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\Projects\\Ongoing\\FOMO\Results\\Steffen\\LengthOfSimulation\\'
        filename = 'convergence_do_nothing_'+str(instance)
        extensions = ['.png','.pdf']
        for ext in extensions:
            plt.savefig(location+filename+ext, dpi=150)
        #plt.show()
        
        # ax = df_merged[['starv','cong']].plot.area(title=instance)
        # ax.set_xlabel("time in days")
        # ax.set_ylabel("lost trips (%)")
        # plt.show()

    
    plot_lost_trips_over_time(metric,instance)

if __name__ == "__main__":
    
    for instance in ['OS_W31','TD_W34']: 
        main(instance)
