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

# TO DO: update to new setup

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
NUM_DAYS = 3*7
DURATION = timeInMinutes(hours=24*NUM_DAYS)
WEEK = 34

instances = ['EH_W22','TD_W34','BG_W25','OS_W31']    

if __name__ == "__main__":
    
    for INSTANCE in instances:
        print(INSTANCE)
        state = init_state.read_initial_state("instances/"+INSTANCE)
        
        total_num_locks = 0
        for station_id,station in state.stations.items():
            total_num_locks += station.capacity
        
        print(total_num_locks/2)
