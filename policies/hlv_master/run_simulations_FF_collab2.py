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
        # 'Base': [(policies.hlv_master.BS_PILOT(), True), (policies.hlv_master.BS_PILOT_FF(), False)],
        'Collab2': [(policies.hlv_master.SB_Collab2(), True), (policies.hlv_master.FF_Collab2(), False)],
        # 'Collab3': [(policies.hlv_master.Collab3(), None)],
        # 'Collab4': [(policies.hlv_master.Collab4(), None)]
                   }
  
    start_time = time.time()
    
    test_policies(list_of_seeds=SEEDS_LIST, policy_dict=policy_dict)

    duration = time.time() - start_time
    print("Running time: ", str(duration))