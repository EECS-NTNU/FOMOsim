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
        # 'SB_base': (policies.hlv_master.BS_PILOT(), True),
        'FF_base': (policies.hlv_master.BS_PILOT_FF(), False),
        # 'SB_Collab2': (policies.hlv_master.SB_Collab2(), True),
        # 'FF_Collab2': (policies.hlv_master.FF_Collab2(), False),
        # 'Collab3': (policies.hlv_master.Collab3(), None),
        # 'Collab4': (policies.hlv_master.Collab4(), None),
        # f'DoNothing_res{RESOLUTION}_rad{RADIUS}': (policies.DoNothing(), None)
        }
  
    start_time = time.time()
    
    # test_policies(list_of_seeds=SEEDS_LIST, policy_dict=policy_dict)
    crit_set = {'a': [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                'b': [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
                'c': [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                'd': [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                'e': [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
                'f': [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                'g': [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]],
                'h': [[0.025, 0.06, 0.1, 0.165, 0.25, 0.4]],
                'i': [[0.165, 0.25, 0.4, 0.025, 0.06, 0.1]],
                'j': [[0.25, 0.1, 0.025, 0.4, 0.06, 0.165]],
                'k': [[0.4, 0.165, 0.1, 0.06, 0.25, 0.025]],
                'l': [[0.025, 0.4, 0.1, 0.25, 0.165, 0.06]],
                'm': [[0.025, 0.165, 0.25, 0.1, 0.4, 0.06]],
                'n': [[0.025, 0.06, 0.4, 0.25, 0.1, 0.165]],
                'o': [[0.25, 0.06, 0.025, 0.165, 0.4, 0.1]],
                'p': [[0.06, 0.1, 0.165, 0.4, 0.25, 0.025]],
                'q': [[0.1, 0.165, 0.25, 0.06, 0.025, 0.4]],
                'r': [[0.165, 0.4, 0.025, 0.06, 0.1, 0.25]],
                's': [[0.4, 0.025, 0.165, 0.1, 0.06, 0.25]],
                't': [[0.06, 0.4, 0.25, 0.1, 0.025, 0.165]],
                'u': [[0.1, 0.025, 0.165, 0.06, 0.4, 0.25]],
                'v': [[0.165, 0.025, 0.4, 0.1, 0.25, 0.06]],
                'w': [[0.25, 0.1, 0.025, 0.06, 0.165, 0.4]],
                'x': [[0.4, 0.25, 0.025, 0.165, 0.06, 0.1]],
                'y': [[0.025, 0.165, 0.1, 0.4, 0.06, 0.25]],
                'aa': [[0.06, 0.1, 0.4, 0.025, 0.165, 0.25]],
                'ab': [[0.06, 0.165, 0.025, 0.25, 0.1, 0.4]],
                'ac': [[0.025, 0.1, 0.06, 0.25, 0.4, 0.165]],
                'ad': [[0.06, 0.25, 0.025, 0.4, 0.165, 0.1]]}
    
    test_criticality_weights(list_of_seeds=SEEDS_LIST, dict_of_criticality_weights=crit_set, policy_name='Base')

    duration = time.time() - start_time
    print("Running time: ", str(duration))