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
        policy_dict = { 'Base': [(policies.hlv_master.BS_PILOT(discounting_factor= discount_factor), True), 
                                (policies.hlv_master.BS_PILOT_FF(discounting_factor= discount_factor), False)],
                        'Collab2': [(policies.hlv_master.SB_Collab2(discounting_factor= discount_factor), True), 
                                    (policies.hlv_master.FF_Collab2(discounting_factor= discount_factor), False)],
                        'Collab3': [(policies.hlv_master.Collab3(discounting_factor= discount_factor), None)],
                        'Collab4': [(policies.hlv_master.Collab4(discounting_factor= discount_factor), None)]
                        }
        test_resolutions(policy_dict[policy_name], filename, policy_name, list_of_seeds)

def test_scenarios(list_of_seeds, list_of_scenarios, policy_name):
    for scenario in list_of_scenarios:
        filename = f'num_scenarios_{scenario}_{policy_name}.csv'
        policy_dict = { 'Base': [(policies.hlv_master.BS_PILOT(number_of_scenarios= scenario), True), 
                                (policies.hlv_master.BS_PILOT_FF(number_of_scenarios= scenario), False)],
                        'Collab2': [(policies.hlv_master.SB_Collab2(number_of_scenarios= scenario), True), 
                                    (policies.hlv_master.FF_Collab2(number_of_scenarios= scenario), False)],
                        'Collab3': [(policies.hlv_master.Collab3(number_of_scenarios= scenario), None)],
                        'Collab4': [(policies.hlv_master.Collab4(number_of_scenarios= scenario), None)]
                        }
        test_resolutions(policy_dict[policy_name], filename, policy_name, list_of_seeds)

def test_alpha_beta(list_of_seeds, alpha, list_of_betas, policy_name):
    for beta in list_of_betas:
        filename = f'branching_a_{alpha}_b_{beta}_{policy_name}.csv'
        policy_dict = { 'Base': [(policies.hlv_master.BS_PILOT(max_depth = alpha, number_of_successors = beta), True), 
                                (policies.hlv_master.BS_PILOT_FF(max_depth = alpha, number_of_successors = beta), False)],
                        'Collab2': [(policies.hlv_master.SB_Collab2(max_depth = alpha, number_of_successors = beta), True), 
                                    (policies.hlv_master.FF_Collab2(max_depth = alpha, number_of_successors = beta), False)],
                        'Collab3': [(policies.hlv_master.Collab3(max_depth = alpha, number_of_successors = beta), None)],
                        'Collab4': [(policies.hlv_master.Collab4(max_depth = alpha, number_of_successors = beta), None)]
                        }
        test_resolutions(policy_dict[policy_name], filename, policy_name, list_of_seeds)

def test_swap_threshold(list_of_seeds, list_of_swap_thresholds, policy_name):
    for threshold in list_of_swap_thresholds:
        filename = f'swap_threshold_{threshold}_{policy_name}.csv'
        policy_dict = { 'Base': [(policies.hlv_master.BS_PILOT(swap_threshold = threshold), True), 
                                (policies.hlv_master.BS_PILOT_FF(swap_threshold = threshold), False)],
                        'Collab2': [(policies.hlv_master.SB_Collab2(swap_threshold = threshold), True), 
                                    (policies.hlv_master.FF_Collab2(swap_threshold = threshold), False)],
                        'Collab3': [(policies.hlv_master.Collab3(swap_threshold = threshold), None)],
                        'Collab4': [(policies.hlv_master.Collab4(swap_threshold = threshold), None)]
                        }
        test_resolutions(policy_dict[policy_name], filename, policy_name, list_of_seeds)

def test_num_clusters(list_of_seeds, list_of_num_clusters, policy_name):
    for num in list_of_num_clusters:
        filename = f'num_clusters_{num}_{policy_name}.csv'
        policy_dict = { 'Base': [(policies.hlv_master.BS_PILOT(), True), 
                                (policies.hlv_master.BS_PILOT_FF(num_clusters = num), False)],
                        'Collab2': [(policies.hlv_master.SB_Collab2(), True), 
                                    (policies.hlv_master.FF_Collab2(num_clusters = num), False)],
                        'Collab3': [(policies.hlv_master.Collab3(num_clusters = num), None)],
                        'Collab4': [(policies.hlv_master.Collab4(num_clusters = num), None)]
                        }
        test_resolutions(policy_dict[policy_name], filename, policy_name, list_of_seeds)

def test_starvation_congestion(list_of_seeds, list_of_congestions, list_of_starvations, policy_name):
    for congestion in list_of_congestions:
        for starvation in list_of_starvations:
            filename = f'starvation_{starvation}_congestion_{congestion}_{policy_name}.csv'
            policy_dict = { 'Base': [(policies.hlv_master.BS_PILOT(congestion_criteria = congestion, starvation_criteria = starvation), True), 
                                (policies.hlv_master.BS_PILOT_FF(congestion_criteria = congestion, starvation_criteria = starvation), False)],
                        'Collab2': [(policies.hlv_master.SB_Collab2(congestion_criteria = congestion, starvation_criteria = starvation), True), 
                                    (policies.hlv_master.FF_Collab2(congestion_criteria = congestion, starvation_criteria = starvation), False)],
                        'Collab3': [(policies.hlv_master.Collab3(congestion_criteria = congestion, starvation_criteria = starvation), None)],
                        'Collab4': [(policies.hlv_master.Collab4(congestion_criteria = congestion, starvation_criteria = starvation), None)]
                        }
            test_resolutions(policy_dict[policy_name], filename, policy_name, list_of_seeds)

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

def run_tests(number):
    # evaluation_weights = [{
    #           'a': [0.0, 0.0, 1.0],
    #           'b': [0.1, 0.3, 0.6],
    #           'c': [0.25, 0.25, 0.5],
    #           'd': [0.1, 0.6, 0.3],
    #           'e': [0.25, 0.5, 0.25]},
    #           {'f': [0.0, 0.1, 0.0],
    #           'g': [0.3, 0.1, 0.6],
    #           'h': [0.33, 0.33, 0.33],
    #           'i': [0.3, 0.6, 0.1],
    #           'j': [0.5, 0.25, 0.25]},
    #           {'k': [0.67, 0.0, 0.33],
    #           'l': [0.67, 0.33, 0.0],
    #           'm': [0.1, 0.0, 0.0],
    #           'n': [0.8, 0, 0.2]},
    #           {'o': [0.15, 0.1, 0.75],
    #           'p': [0.05, 0.05, 0.9],
    #           'q': [0.2, 0, 0.8],
    #           'r': [0.9, 0, 0.1]},
    #           {'s': [0.95, 0.0, 0.05],
    #           't': [0.85, 0, 0.15],
    #           'v': [0.5, 0, 0.5],
    #           'u': [0, 0, 1]}
    # ]
    # test_evaluation_sets(list_of_seeds=SEEDS_LIST, dict_of_evaluation_sets=evaluation_weights[number], policy_name='Base')

    # discount_factors = [
    #                         [1.0, 0.1],
    #                         [0.2, 0.3],
    #                         [0.4, 0.5],
    #                         [0.6, 0.7],
    #                         [0.8, 0.9]
    #                     ]
    # test_discount_factor(list_of_seeds= SEEDS_LIST, list_of_discout_factors = discount_factors[number], policy_name='Base')
    
    # crit_set = [
    # #             {'a': [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    # #             'b': [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
    # #             'c': [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
    # #             'd': [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
    # #             'e': [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
    # #             'f': [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},

    # #             {'g': [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]],
    # #             'h': [[0.025, 0.06, 0.1, 0.165, 0.25, 0.4]],
    # #             'i': [[0.165, 0.25, 0.4, 0.025, 0.06, 0.1]],
    # #             'j': [[0.25, 0.1, 0.025, 0.4, 0.06, 0.165]],
    # #             'k': [[0.4, 0.165, 0.1, 0.06, 0.25, 0.025]],
    # #             'l': [[0.025, 0.4, 0.1, 0.25, 0.165, 0.06]]},
                
    # #             {'m': [[0.025, 0.165, 0.25, 0.1, 0.4, 0.06]],
    # #             'n': [[0.025, 0.06, 0.4, 0.25, 0.1, 0.165]],
    # #             'o': [[0.25, 0.06, 0.025, 0.165, 0.4, 0.1]],
    # #             'p': [[0.06, 0.1, 0.165, 0.4, 0.25, 0.025]],
    # #             'q': [[0.1, 0.165, 0.25, 0.06, 0.025, 0.4]],
    # #             'r': [[0.165, 0.4, 0.025, 0.06, 0.1, 0.25]]},

    # #             {'s': [[0.4, 0.025, 0.165, 0.1, 0.06, 0.25]],
    # #             't': [[0.06, 0.4, 0.25, 0.1, 0.025, 0.165]],
    # #             'u': [[0.1, 0.025, 0.165, 0.06, 0.4, 0.25]],
    # #             'v': [[0.165, 0.025, 0.4, 0.1, 0.25, 0.06]],
    # #             'w': [[0.25, 0.1, 0.025, 0.06, 0.165, 0.4]],
    # #             'x': [[0.4, 0.25, 0.025, 0.165, 0.06, 0.1]]},

    # #             {'y': [[0.025, 0.165, 0.1, 0.4, 0.06, 0.25]],
    # #             'aa': [[0.06, 0.1, 0.4, 0.025, 0.165, 0.25]],
    # #             'ab': [[0.06, 0.165, 0.025, 0.25, 0.1, 0.4]],
    # #             'ac': [[0.025, 0.1, 0.06, 0.25, 0.4, 0.165]],
    # #             'ad': [[0.06, 0.25, 0.025, 0.4, 0.165, 0.1]],
    # #             'z': [[0.06, 0.4, 0.165, 0.025, 0.25, 0.1]]}
    # ]

    # crit_set = [dict(
    #     a=[[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]], 
    #     b=[[0.3, 0.15, 0.25, 0.2, 0.1, 0]], 
    #     c=[[0.15, 0.3, 0.15, 0.1, 0.1, 0.2]], 
    #     d=[[0.25, 0.25, 0.1, 0.1, 0.2, 0.1]]), # Balanced

    #     dict(
    #     e=[[0.15, 0.4, 0.05, 0.05, 0, 0.35]], 
    #     f=[[0.05, 0.9, 0.05, 0, 0, 0]], 
    #     g=[[0.1, 0.5, 0.1, 0.1, 0.1, 0.1]], 
    #     h=[[0.3, 0.5, 0, 0, 0.2, 0]]), # Long term

    #     dict(
    #     i=[[0.4, 0, 0, 0.1, 0, 0.5]], 
    #     j=[[0.6, 0.05, 0.05, 0.1, 0.05, 0.15]], 
    #     k=[[0.6, 0.1, 0.05, 0.2, 0.05, 0]], 
    #     l=[[0.5, 0.05, 0.1, 0.05, 0.1, 0.2]]), # Short term

    #     dict(
    #     m=[[1, 0, 0, 0, 0, 0]], 
    #     n=[[0, 1, 0, 0, 0, 0]], 
    #     o=[[0, 0, 1, 0, 0, 0]]),

    #     dict(
    #     p=[[0, 0, 0, 1, 0, 0]], 
    #     q=[[0, 0, 0, 0, 1, 0]], 
    #     r=[[0, 0, 0, 0, 0, 1]])] # Extremes

        # s= [[0.05, 0.85, 0.05, 0, 0, 0.05]], # f
        # t=[[0.25, 0.45, 0, 0, 0.2, 0.1]], # h
        # u = [[0.3, 0.45, 0, 0, 0.2, 0.05]], # h
        # v =[[0.5, 0.1, 0.05, 0.2, 0.05, 0.1]], # k
        # w =[[0.55, 0.1, 0.05, 0.2, 0.05, 0.05]], # k
        # x =[[0.45, 0.1, 0.05, 0.2, 0.05, 0.15]], # k
        # y =[[0.6, 0.1, 0.05, 0.15, 0.05, 0.05]], # k
        # z =[[0.6, 0.05, 0.05, 0.15, 0.05, 0.1]] # k
        # Long term

    # test_criticality_weights(list_of_seeds=SEEDS_LIST, dict_of_criticality_weights=crit_set[number], policy_name='Base')

    num_clusters = [[4, 50],
                    [8, 30],
                    [15, 20],
                    [25, 10],
                    [40, 6]
                    ]
    test_num_clusters(list_of_seeds=SEEDS_LIST, list_of_num_clusters=num_clusters[number], policy_name='Base')

    # alphas = [[1,2],
    #           [3,4],
    #           [5],
    #           [6],
    #           [7]]
    # betas =[1,2,3,4,5]
    # for a in alphas[number]:
    #     test_alpha_beta(list_of_seeds=SEEDS_LIST, alpha=a, list_of_betas=betas, policy_name='Base')

    # time_horizons = [[10, 20],
    #                  [30],
    #                  [40],
    #                  [50],
    #                  [60]    
    # ]
    # test_timehorizons(list_of_seeds=SEEDS_LIST, list_of_timehorizons=time_horizons[number], policy_name='Base')

    # num_scenarios = [[1, 10, 120],
    #                  [20, 100],
    #                  [30, 90],
    #                  [40, 80],
    #                  [50, 60, 70]
    #                  ]
    # test_scenarios(list_of_seeds=SEEDS_LIST, list_of_scenarios=num_scenarios[number], policy_name='Base')

    # adj_factors = [[0.25, 0.5, 0.6, 2],
    #                 [0.7, 0.8, 0.9],
    #                 [1, 1.1, 1.2],
    #                 [1.3, 1.4, 1.5],
    #                 [1.6, 1.7, 1.8, 1.9]]
    # test_adjustment_factor(list_of_seeds=SEEDS_LIST, list_of_adjustment_factors=adj_factors[number], policy_name="Collab3")


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


    dict_ev= {
              'a': [0.0, 0.0, 1.0],
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
              'm': [0.1, 0.0, 0.0],
              'n': [0.8, 0, 0.2],
              
              'o': [0.15, 0.1, 0.75],
              'p': [0.05, 0.05, 0.9],
              'q': [0.2, 0, 0.8],
              'r': [0.9, 0, 0.1],
              
              's': [0.95, 0.0, 0.05],
              't': [0.85, 0, 0.15],
              'v': [0.5, 0, 0.5],
              'u': [0, 0, 1]
    }
    test_evaluation_sets(list_of_seeds=SEEDS_LIST, dict_of_evaluation_sets=dict_ev, policy_name='Base')

    duration = time.time() - start_time
    print("Running time: ", str(duration))