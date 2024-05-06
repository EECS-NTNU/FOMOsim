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

def run_simulation(seed, policy, is_SB, filename_sb, filename_ff, target_filename, operator_radius, roaming_radius):
    state = init_state.read_initial_state(sb_jsonFilename = filename_sb, ff_jsonFilename = filename_ff)
    state.set_seed(seed)
    state.roaming_radius = roaming_radius
    vehicles = [policy for i in range(NUM_VEHICLES)]
    for p in vehicles:
        p.operator_radius = operator_radius

    if is_SB is None:
        state.set_vehicles(vehicles)
    elif is_SB:
        state.set_sb_vehicles(vehicles)
    else:
        state.set_ff_vehicles(vehicles)
    
    tstate = target_state.HLVTargetState(target_filename)
    
    dmand = demand.Demand()
    simulator = sim.Simulator(
        initial_state = state,
        target_state = tstate,
        demand = dmand,
        start_time = START_TIME,
        duration = DURATION,
        verbose = True,
    )
    simulator.run()
    return simulator.metrics, seed

def test_policies(list_of_seeds, policy_dict):
    for policy_name in policy_dict:
        res_filename=str(policy_name)+".csv"
        policy, is_SB = policy_dict[policy_name]
        resolutions = [11,10,9,8]
        hex_radiuss = [100,58,22,10]
        roaming_radiuss = [9, 4, 2, 0]
        operator_radiuss = [5, 2, 1, 0]
        
        for i in range(4):
            resolution = resolutions[i]
            hex_radius = hex_radiuss[i]
            roaming_radius = roaming_radiuss[i]
            operator_radius = operator_radiuss[i]

            filename_sb = 'instances/TD_W34'
            filename_ff = f'instances/Ryde/TD_700_res{resolution}_radius{hex_radius}_W3'
            target_filename = f'instances/Ryde/target_states_700_res{resolution}_radius{hex_radius}.json.gz'
            
            test_seeds_mp(list_of_seeds, policy, is_SB, res_filename, policy_name+f'_res{resolution}_rad{hex_radius}', filename_sb, filename_ff, target_filename, operator_radius, roaming_radius)

def test_seeds_mp(list_of_seeds, policy, is_SB, results_filename, policy_name, filename_sb, filename_ff, target_filename, operator_radius, roaming_radius): #change duration and number of vehicles HERE!    
    args = [(seed, policy, is_SB, filename_sb, filename_ff, target_filename, operator_radius, roaming_radius) for seed in list_of_seeds]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(run_simulation, args)
    
    append = False
    for simulator_metrics, seed in results:
        write_sim_results_to_file(results_filename, simulator_metrics, seed, DURATION, policy_name, append=append)

        filename_time = "sol_time_"+results_filename
        write_sol_time_to_file(filename_time, simulator_metrics, policy_name, append)
        append = True

        # TODO alle som skrives i write_sim_results_to_file
        visualize(results_filename, policy_name, simulator_metrics, 'failed events')
        visualize(results_filename, policy_name, simulator_metrics, 'trips')
        visualize(results_filename, policy_name, simulator_metrics, 'bike starvations')
        visualize(results_filename, policy_name, simulator_metrics, 'escooter starvations')
        visualize(results_filename, policy_name, simulator_metrics, 'battery starvations')
        visualize(results_filename, policy_name, simulator_metrics, 'battery violations')
        visualize(results_filename, policy_name, simulator_metrics, 'starvations')
        visualize(results_filename, policy_name, simulator_metrics, 'long congestions')

        # TODO fix
        write_parameters_to_file('parameters_' + results_filename, policy, policy_name, NUM_VEHICLES, DURATION)
    
    # visualize_aggregated_results(filename, policy_name)

if __name__ == "__main__":
    policy_dict = {
        'SB_base': (policies.hlv_master.BS_PILOT(), True),
        'FF_base': (policies.hlv_master.BS_PILOT_FF(), False),
        'SB_Collab2': (policies.hlv_master.SB_Collab2(), True),
        'FF_Collab2': (policies.hlv_master.FF_Collab2(), False),
        'Collab3': (policies.hlv_master.Collab3(), None),
        'Collab4': (policies.hlv_master.Collab4(), None)
                   }
  
    list_of_seeds = SEEDS_LIST
  
    start_time = time.time()
    
    test_policies(list_of_seeds=list_of_seeds, policy_dict=policy_dict)

    duration = time.time() - start_time
    print("Running time: ", str(duration))