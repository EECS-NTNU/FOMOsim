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

def run_simulation(seed, policies, filename_sb, filename_ff, target_filename, operator_radius, roaming_radius, num_vehicles):
    state = init_state.read_initial_state(sb_jsonFilename = filename_sb, ff_jsonFilename = filename_ff)
    state.set_seed(seed)
    state.roaming_radius = roaming_radius
    policies *= num_vehicles
    sb_vehicles = [policy for policy, is_SB in policies if is_SB == True]
    ff_vehicles = [policy for policy, is_SB in policies if is_SB == False]
    all_vehicles = [policy for policy, is_SB in policies if is_SB == None]

    for p in sb_vehicles:
        p.operator_radius = operator_radius
    for p in ff_vehicles:
        p.operator_radius = operator_radius
    for p in all_vehicles:
        p.operator_radius = operator_radius
    
    state.set_vehicles(all_vehicles)
    state.set_sb_vehicles(sb_vehicles)
    state.set_ff_vehicles(ff_vehicles)

    print(len(state.get_vehicles()))
    
    tstate = target_state.HLVTargetState(target_filename)
    tstate.set_target_states(state)
    
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
        policies = policy_dict[policy_name]
        test_resolutions(policies, res_filename, policy_name, list_of_seeds)

def test_timehorizons(list_of_seeds, list_of_timehorizons, policy_name):
    for timehorizon in list_of_timehorizons:
        filename = f'timehorizon_{timehorizon}_{policy_name}.csv'
        if policy_name == 'FF_base':
            policies = [policies.BS_PILOT_FF(time_horizon= timehorizon)]
            is_SB = [False]
        elif policy_name == 'SB_base':
            policies = [policies.BS_PILOT(time_horizon=timehorizon)]
            is_SB = [True]
        elif policy_name == 'FF_collab2':
            policies = [policies.FF_Collab2(time_horizon=timehorizon), policies.SB_Collab2(time_horizon=timehorizon)]
            is_SB = [False, True]
        elif policy_name == 'Collab3':
            policies = [policies.Collab3(time_horizon=timehorizon)]
            is_SB = [None]
        elif policy_name == 'Collab4':
            policies = [policies.Collab4(time_horizon=timehorizon)]
            is_SB = [None]
        test_resolutions(policies, filename, policy_name, list_of_seeds)

def test_criticality_weights(list_of_seeds, dict_of_criticality_weights, policy_name):
    for letter, criticality_set in dict_of_criticality_weights.items():
        filename = f'criticality_weights_{letter}_{policy_name}.csv'
        policy_dict = { 'Base': [(policies.hlv_master.BS_PILOT(criticality_weights_set = criticality_set), True), 
                                (policies.hlv_master.BS_PILOT_FF(criticality_weights_set = criticality_set), False)],
                        'Collab2': [(policies.hlv_master.SB_Collab2(criticality_weights_set = criticality_set), True), 
                                    (policies.hlv_master.FF_Collab2(criticality_weights_set = criticality_set), False)],
                        'Collab3': [(policies.hlv_master.Collab3(criticality_weights_set_ff = criticality_set), None)],
                        'Collab4': [(policies.hlv_master.Collab4(criticality_weights_set_ff = criticality_set), None)]
                        }
        test_resolutions(policy_dict[policy_name], filename, policy_name, list_of_seeds)

def test_criticality_weights_collab34(list_of_seeds, list_of_criticality_weights_sb, list_of_criticality_weights_ff, policy_name):
    for letter1, criticality_set_sb in list_of_criticality_weights_sb:
        for letter2, criticality_set_ff in list_of_criticality_weights_ff:
            filename = f'criticality_weights_{letter1}_{letter2}_{policy_name}.csv'
            policy_dict = { 'Collab3': [(policies.hlv_master.Collab3(criticality_weights_set_sb = criticality_set_sb, criticality_weights_set_ff = criticality_set_ff), None)],
                            'Collab4': [(policies.hlv_master.Collab4(criticality_weights_set_sb = criticality_set_sb, criticality_weights_set_ff = criticality_set_ff), None)]
                            }
            test_resolutions(policy_dict[policy_name], filename, policy_name, list_of_seeds)

def test_adjustment_factor(list_of_seeds, list_of_adjustment_factors, policy_name):
    for factor in list_of_adjustment_factors:
        filename = f'adjustment_factor_{factor}_{policy_name}.csv'
        if policy_name == 'Collab3':
            policy = policies.Collab3(adjusting_criticality=factor)
            is_SB = None
        elif policy_name == 'Collab4':
            policy = policies.Collab4(adjusting_criticality=factor)
            is_SB = None
        test_resolutions(policy, is_SB, filename, policy_name, list_of_seeds)

def test_evaluation_sets(list_of_seeds, dict_of_evaluation_sets, policy_name):
    for letter, evaluation_set in dict_of_evaluation_sets.items():
        filename = f'evaluation_set_{letter}_{policy_name}.csv'
        policy_dict = { 'Base': [(policies.hlv_master.BS_PILOT(evaluation_weights = evaluation_set), True), 
                                (policies.hlv_master.BS_PILOT_FF(evaluation_weights = evaluation_set), False)],
                        'Collab2': [(policies.hlv_master.SB_Collab2(evaluation_weights = evaluation_set), True), 
                                    (policies.hlv_master.FF_Collab2(evaluation_weights = evaluation_set), False)],
                        'Collab3': [(policies.hlv_master.Collab3(evaluation_weights = evaluation_set), None)],
                        'Collab4': [(policies.hlv_master.Collab4(evaluation_weights = evaluation_set), None)]
                        }
        test_resolutions(policy_dict[policy_name], filename, policy_name, list_of_seeds)

def test_discount_factor(list_of_seeds, list_of_discout_factors, policy_name):
    for discount_factor in list_of_discout_factors:
        filename = f'discount_factor_{discount_factor}_{policy_name}.csv'
        if policy_name == 'FF_base':
            policy = policies.BS_PILOT_FF(discounting_factor= discount_factor)
            is_SB = True
        elif policy_name == 'SB_base':
            policy = policies.BS_PILOT(discounting_factor=discount_factor)
            is_SB = False
        elif policy_name == 'FF_collab2':
            policy = policies.FF_Collab2(discounting_factor=discount_factor)
            is_SB = True
        elif policy_name == 'SB_collab2':
            policy = policies.SB_Collab2(discounting_factor=discount_factor)
            is_SB = False
        elif policy_name == 'Collab3':
            policy = policies.Collab3(discounting_factor=discount_factor)
            is_SB = None
        elif policy_name == 'Collab4':
            policy = policies.Collab4(discounting_factor=discount_factor)
            is_SB = None
        test_resolutions(policy, is_SB, filename, policy_name, list_of_seeds)

def test_scenarios(list_of_seeds, list_of_scenarios, policy_name):
    for scenario in list_of_scenarios:
        filename = f'num_scenarios_{scenario}_{policy_name}.csv'
        if policy_name == 'FF_base':
            policy = policies.BS_PILOT_FF(number_of_scenarios= scenario)
            is_SB = True
        elif policy_name == 'SB_base':
            policy = policies.BS_PILOT(number_of_scenarios=scenario)
            is_SB = False
        elif policy_name == 'FF_collab2':
            policy = policies.FF_Collab2(number_of_scenarios=scenario)
            is_SB = True
        elif policy_name == 'SB_collab2':
            policy = policies.SB_Collab2(number_of_scenarios=scenario)
            is_SB = False
        elif policy_name == 'Collab3':
            policy = policies.Collab3(number_of_scenarios=scenario)
            is_SB = None
        elif policy_name == 'Collab4':
            policy = policies.Collab4(number_of_scenarios=scenario)
            is_SB = None
        test_resolutions(policy, is_SB, filename, policy_name, list_of_seeds)

def test_alpha_beta(list_of_seeds, alpha, list_of_betas, policy_name):
    for beta in list_of_betas:
        filename = f'branching_a_{alpha}_b_{beta}_{policy_name}.csv'
        if policy_name == 'FF_base':
            policy = policies.BS_PILOT_FF(max_depth = alpha, number_of_successors = beta)
            is_SB = True
        elif policy_name == 'SB_base':
            policy = policies.BS_PILOT(max_depth = alpha, number_of_successors = beta)
            is_SB = False
        elif policy_name == 'FF_collab2':
            policy = policies.FF_Collab2(max_depth = alpha, number_of_successors = beta)
            is_SB = True
        elif policy_name == 'SB_collab2':
            policy = policies.SB_Collab2(max_depth = alpha, number_of_successors = beta)
            is_SB = False
        elif policy_name == 'Collab3':
            policy = policies.Collab3(max_depth = alpha, number_of_successors = beta)
            is_SB = None
        elif policy_name == 'Collab4':
            policy = policies.Collab4(max_depth = alpha, number_of_successors = beta)
            is_SB = None
        test_resolutions(policy, is_SB, filename, policy_name, list_of_seeds)

def test_swap_threshold(list_of_seeds, list_of_swap_thresholds, policy_name):
    for threshold in list_of_swap_thresholds:
        filename = f'swap_threshold_{threshold}_{policy_name}.csv'
        if policy_name == 'FF_base':
            policy = policies.BS_PILOT_FF(swap_threshold = threshold)
            is_SB = True
        elif policy_name == 'SB_base':
            policy = policies.BS_PILOT(swap_threshold = threshold)
            is_SB = False
        elif policy_name == 'FF_collab2':
            policy = policies.FF_Collab2(swap_threshold = threshold)
            is_SB = True
        elif policy_name == 'SB_collab2':
            policy = policies.SB_Collab2(swap_threshold = threshold)
            is_SB = False
        elif policy_name == 'Collab3':
            policy = policies.Collab3(swap_threshold = threshold)
            is_SB = None
        elif policy_name == 'Collab4':
            policy = policies.Collab4(swap_threshold = threshold)
            is_SB = None
        test_resolutions(policy, is_SB, filename, policy_name, list_of_seeds)

def test_num_clusters(list_of_seeds, list_of_num_clusters, policy_name):
    for num in list_of_num_clusters:
        filename = f'num_clusters_{num}_{policy_name}.csv'
        if policy_name == 'FF_base':
            policy = policies.BS_PILOT_FF(num_clusters = num)
            is_SB = True
        elif policy_name == 'FF_collab2':
            policy = policies.FF_Collab2(num_clusters = num)
            is_SB = True
        elif policy_name == 'Collab3':
            policy = policies.Collab3(num_clusters = num)
            is_SB = None
        elif policy_name == 'Collab4':
            policy = policies.Collab4(num_clusters = num)
            is_SB = None
        test_resolutions(policy, is_SB, filename, policy_name, list_of_seeds)

def test_starvation_congestion(list_of_seeds, list_of_congestions, list_of_starvations, policy_name):
    for congestion in list_of_congestions:
        for starvation in list_of_starvations:
            filename = f'starvation_{starvation}_congestion_{congestion}_{policy_name}.csv'
            if policy_name == 'SB_base':
                policy = policies.BS_PILOT(congestion_criteria = congestion, starvation_criteria = starvation)
                is_SB = True
            elif policy_name == 'SB_collab2':
                policy = policies.SB_Collab2(congestion_criteria = congestion, starvation_criteria = starvation)
                is_SB = True
            elif policy_name == 'Collab3':
                policy = policies.Collab3(congestion_criteria = congestion, starvation_criteria = starvation)
                is_SB = None
            elif policy_name == 'Collab4':
                policy = policies.Collab4(congestion_criteria = congestion, starvation_criteria = starvation)
                is_SB = None
            test_resolutions(policy, is_SB, filename, policy_name, list_of_seeds)

def test_resolutions(policies, res_filename, policy_name, list_of_seeds):
    resolutions = [10] #[11, 10, 9]
    hex_radiuss = [58] #[100, 58, 22]
    roaming_radiuss = [2] #[9, 3, 1]
    operator_radiuss = [1] #[4, 1, 0]
    
    for i in range(len(resolutions)):
        resolution = resolutions[i]
        hex_radius = hex_radiuss[i]
        roaming_radius = roaming_radiuss[i]
        operator_radius = operator_radiuss[i]

        filename_sb = 'instances/TD_W34'
        filename_ff = f'instances/Ryde/TD_700_res{resolution}_radius{hex_radius}_W3'
        target_filename = f'instances/Ryde/target_states_700_res{resolution}_radius{hex_radius}.json.gz'
        
        test_seeds_mp(list_of_seeds, policies, res_filename, policy_name+f'_res{resolution}_rad{hex_radius}', filename_sb, filename_ff, target_filename, operator_radius, roaming_radius)

def test_seeds_mp(list_of_seeds, policies, results_filename, policy_name, filename_sb, filename_ff, target_filename, operator_radius, roaming_radius, num_vehicles = NUM_VEHICLES): #change duration and number of vehicles HERE!    
    args = [(seed, policies, filename_sb, filename_ff, target_filename, operator_radius, roaming_radius, num_vehicles) for seed in list_of_seeds]
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
        write_parameters_to_file('parameters_' + results_filename, policies, policy_name, NUM_VEHICLES, DURATION)
    
    # visualize_aggregated_results(filename, policy_name)

if __name__ == "__main__":
    policy_dict = {
        'Base': [(policies.hlv_master.BS_PILOT(), True), (policies.hlv_master.BS_PILOT_FF(), False)],
        'Collab2': [(policies.hlv_master.SB_Collab2(), True), (policies.hlv_master.FF_Collab2(), False)],
        'Collab3': [(policies.hlv_master.Collab3(), None)],
        'Collab4': [(policies.hlv_master.Collab4(), None)]
                   }
  
    list_of_seeds = SEEDS_LIST
  
    start_time = time.time()
    
    test_policies(list_of_seeds=list_of_seeds, policy_dict=policy_dict)

    duration = time.time() - start_time
    print("Running time: ", str(duration))