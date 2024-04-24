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

def run_simulation(seed, policy, is_SB):
    state = init_state.read_initial_state(sb_jsonFilename = SB_INSTANCE_FILE, ff_jsonFilename = FF_INSTANCE_FILE)
    state.set_seed(seed)
    vehicles = [policy for i in range(NUM_VEHICLES)]

    if is_SB is None:
        state.set_vehicles(vehicles)
    elif is_SB:
        state.set_sb_vehicles(vehicles)
    else:
        state.set_ff_vehicles(vehicles)
    
    tstate = target_state.HLVTargetState(FF_TARGET_STATE_FILE)
    
    dmand = demand.Demand()
    simulator = sim.Simulator(
        initial_state = state,
        target_state = tstate,
        demand = dmand,
        start_time = START_TIME,
        duration = DURATION,
        verbose = False,
    )
    simulator.run()
    return simulator.metrics, seed

def test_policies(list_of_seeds, policy_dict):
    for policy_name in policy_dict:
        filename=str(policy_name)+".csv"
        policy, is_SB = policy_dict[policy_name]
        test_seeds_mp(list_of_seeds, policy, is_SB, filename, policy_name)

def test_seeds_mp(list_of_seeds, policy, is_SB, filename, policy_name): #change duration and number of vehicles HERE!
    args = [(seed, policy, is_SB) for seed in list_of_seeds]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(run_simulation, args)
    
    append = False
    for simulator_metrics, seed in results:
        write_sim_results_to_file(filename, simulator_metrics, seed, DURATION, policy_name, append=append)

        filename_time = "sol_time_"+filename
        write_sol_time_to_file(filename_time, simulator_metrics, policy_name, append)
        append = True

        # TODO alle som skrives i write_sim_results_to_file
        visualize(filename, policy_name, simulator_metrics, 'failed events')
        visualize(filename, policy_name, simulator_metrics, 'trips')
        visualize(filename, policy_name, simulator_metrics, 'bike starvations')
        visualize(filename, policy_name, simulator_metrics, 'escooter starvations')
        visualize(filename, policy_name, simulator_metrics, 'battery starvations')
        visualize(filename, policy_name, simulator_metrics, 'battery violations')
        visualize(filename, policy_name, simulator_metrics, 'starvations')
        visualize(filename, policy_name, simulator_metrics, 'long congestions')

        # TODO fix
        write_parameters_to_file('parameters_' + filename, policy, policy_name, NUM_VEHICLES, DURATION)
    
    # visualize_aggregated_results(filename, policy_name)

if __name__ == "__main__":
    
    policy_dict = {
        # 'SB_base': (policies.hlv_master.BS_PILOT(), True),
        'FF_base_skriv_tid': (policies.hlv_master.BS_PILOT_FF(), False),
        # 'SB_Collab2': (policies.hlv_master.SB_Collab2(), True),
        # 'FF_Collab2': (policies.hlv_master.FF_Collab2(), False),
        # 'Collab3': (policies.hlv_master.Collab3(), None),
        # 'Collab4': (policies.hlv_master.Collab4(), None)
        }
  
    start_time = time.time()
    
    test_policies(list_of_seeds=SEEDS_LIST, policy_dict=policy_dict)

    duration = time.time() - start_time
    print("Running time: ", str(duration))