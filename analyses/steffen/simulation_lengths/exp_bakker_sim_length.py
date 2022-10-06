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
NUM_DAYS = 3*7
DURATION = timeInMinutes(hours=24*NUM_DAYS)
WEEK = 34
instance = 'Oslo'

def main(instance):

    ###############################################################################
    # Get initial state

    tstate = target_state.evenly_distributed_target_state
    
    if instance == 'Oslo':
        url_link = 'https://data.urbansharing.com/oslobysykkel.no/trips/v1/'
    elif instance == 'Trondheim':
        url_link = "https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/"
    elif instance == "Bergen":
        url_link = "https://data.urbansharing.com/bergenbysykkel.no/trips/v1/"
    elif instance =="Edinburgh":
        url_link = "https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/"

    state = init_state.get_initial_state(source=init_state.cityBike,
                                         url=url_link,
                                         week=WEEK, random_seed=0,
                                         target_state=tstate,
                                         )

    ###############################################################################
    # Set up policy

    policy = policies.DoNothing()

    ###############################################################################
    # Set up simulator

    simulator = sim.Simulator(
        initial_state = state,
        policy = policy,
        start_time = START_TIME,
        duration = DURATION,
        verbose = True,
    )

    ###############################################################################
    # Run simulator

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
        df_merged = df_merged.set_index('time')
        ax = df_merged[['starv','cong']].plot.area(title=instance)
        ax.set_xlabel("time in days")
        ax.set_ylabel("lost trips (%)")
        plt.show()

    
    plot_lost_trips_over_time(metric,instance)

if __name__ == "__main__":
    
    for instance in ['Oslo','Trondheim']: #'Edinburgh','Bergen'
        main(instance)
