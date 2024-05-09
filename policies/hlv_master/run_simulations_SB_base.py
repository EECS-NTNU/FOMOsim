# -*- coding: utf-8 -*-

######################################################
import os 
import sys
from pathlib import Path
 
path = Path(__file__).parents[2]        # The path seems to be correct either way, sys.path.insert makes the difference
os.chdir(path) 
sys.path.insert(0, '') #make sure the modules are found in the new working directory
################################################################

import init_state
import target_state 
import policies
import sim
import demand
from helpers import timeInMinutes
from settings import *
import time
import multiprocessing as mp
from manage_results import *
from run_simulations import *

if __name__ == "__main__":
    
    policy_dict = {
        'SB_base': (policies.hlv_master.BS_PILOT(), True),
        # 'FF_base': (policies.hlv_master.BS_PILOT_FF(), False),
        # 'SB_Collab2': (policies.hlv_master.SB_Collab2(), True),
        # 'FF_Collab2': (policies.hlv_master.FF_Collab2(), False),
        # 'Collab3': (policies.hlv_master.Collab3(), None),
        # 'Collab4': (policies.hlv_master.Collab4(), None)
        }
  
    start_time = time.time()
    
    # test_policies(list_of_seeds=SEEDS_LIST, policy_dict=policy_dict)
    dict_ev= {'a': [0.0, 0.0, 1.0],
              'b': [0.1, 0.3, 0.6],
              'c': [0.25, 0.25, 0.5],
              'd': [0.1, 0.6, 0.3],
              'e': [0.25, 0.5, 0.25],
              'f': [0.0, 0.1, 0.0],
              'g': [0.3, 0.1, 0.6],
              'h': [0.33, 0.33, 0.33],
              'i': [0.3, 0.6, 0.1],
              'j': [0.5, 0.25, 0.25],
              'k': [0.67, 0.0, 0.33],
              'l': [0.67, 0.33, 0.0],
              'm': [0.1, 0.0, 0.0]}
    
    test_evaluation_sets(list_of_seeds=SEEDS_LIST, dict_of_evaluation_sets=dict_ev, policy_name='Base')

    duration = time.time() - start_time
    print("Running time: ", str(duration))