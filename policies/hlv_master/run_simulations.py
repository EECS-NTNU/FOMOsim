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

def run_simulation(seed, policy, is_SB, duration= DURATION, num_vehicles= NUM_VEHICLES, results=None, INSTANCE=None):

    #change common parameters for the different simulations here:
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=duration)

    INSTANCE = SB_INSTANCE_FILE
    
    ###############################################################
    
    # state = init_state.read_initial_state("instances/ebike/"+INSTANCE)
    state = init_state.read_initial_state(sb_jsonFilename = "instances/"+INSTANCE, ff_jsonFilename="instances/Ryde/TD_W19_test_W3_NEW")
    state.set_seed(seed)
    vehicles = [policy for i in range(num_vehicles)]

    if is_SB is None:
        state.set_vehicles(vehicles)
    elif is_SB:
        state.set_sb_vehicles(vehicles)
    else:
        state.set_ff_vehicles(vehicles)
    
    tstate = target_state.HLVTargetState(
            'instances/Ryde/FINAL_target_state_1066_NEW.json.gz'
            )
    
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
    results.append(simulator.metrics)
    return simulator


def test_policies(list_of_seeds, policy_dict, num_vehicles, duration):
    for policy_name in policy_dict:
        filename=str(policy_name)+".csv"
        policy, is_SB = policy_dict[policy_name]
        # test_seeds_pool_executor(list_of_seeds, policy_dict[policy], filename, num_vehicles, duration)
        test_seeds_mp(list_of_seeds, policy, is_SB, filename, num_vehicles, duration)

def test_timehorizons(list_of_seeds, list_of_timehorizons, is_SB):
    for horizon in list_of_timehorizons:
        filename = "time_horizon_"+str(horizon)+".csv"
        policy=policies.hlv_master.BS_PILOT(time_horizon=horizon, max_depth=5, number_of_successors=7)
        test_seeds_mp(list_of_seeds, policy, is_SB, filename)

def test_criticality_weights(list_of_seeds, criticality_weights_dict, is_SB):
    for set in criticality_weights_dict:
        filename= "crit_set_"+str(set)+".csv"
        policy=policies.hlv_master.BS_PILOT(criticality_weights_set=criticality_weights_dict[set])
        test_seeds_mp(list_of_seeds, policy, is_SB, filename)

def test_evaluation_weights(list_of_seeds, evaluation_weights_dict, is_SB):
     for set in evaluation_weights_dict:
        filename= "evaluation_set_"+str(set)+".csv"
        policy=policies.hlv_master.BS_PILOT(evaluation_weights=evaluation_weights_dict[set])
        test_seeds_mp(list_of_seeds, policy, is_SB, filename)

def test_discounting_factors(list_of_seeds, list_of_factors, is_SB):
     for factor in list_of_factors:
        filename= "discounting_"+str(factor)+".csv"
        policy=policies.hlv_master.BS_PILOT(discounting_factor=factor)
        test_seeds_mp(list_of_seeds, policy, is_SB, filename)

def test_overflow_starvation(list_of_seeds, list_of_overflow, list_of_starvation, is_SB):
     for factor in list_of_overflow:
        for factor2 in list_of_starvation:
            filename= "overflow_starvation_"+str(factor)+str(factor2)+".csv"
            policy=policies.hlv_master.BS_PILOT(overflow_criteria=factor, starvation_criteria = factor2)
            test_seeds_mp(list_of_seeds, policy, is_SB, filename)

def test_alpha_beta(list_of_seeds, alpha, beta_list, is_SB):
     for beta in beta_list:
        filename= "branching_a_"+str(alpha)+"_b_"+str(beta)+".csv"
        policy=policies.hlv_master.BS_PILOT(max_depth=alpha, number_of_successors=beta, time_horizon=60)
        test_seeds_mp(list_of_seeds, policy, is_SB, filename)

def test_number_of_scenarios(list_of_seeds, scenario_list, is_SB):
     for number in scenario_list:
        filename= "num_scenarios_"+str(number)+".csv"
        policy=policies.hlv_master.BS_PILOT(number_of_scenarios=number, max_depth=2, number_of_successors=7)
        test_seeds_mp(list_of_seeds, policy, is_SB, filename)

def test_upper_threshold(list_of_seeds, upper_thresholds, is_SB):
    for threshold in upper_thresholds:
        filename= "upper_threshold_"+str(threshold)+".csv"
        policy=policies.hlv_master.BS_PILOT(upper_thresholds=threshold)
        test_seeds_mp(list_of_seeds, policy, is_SB, filename)

def test_num_vehicles(list_of_seeds, vehicles_list, is_SB):
     for v in vehicles_list:
        filename= "num_vehicles_"+str(v)+"V.csv"
        policy=policies.hlv_master.BS_PILOT(max_depth=2, number_of_successors=5, time_horizon=40)
        # policy = policies.GreedyPolicy()
        # policy= policies.inngjerdingen_moeller.GreedyPolicyNeighborhoodInteraction()
        test_seeds_mp(list_of_seeds, policy, is_SB, filename, num_vehicles=v)

def test_instances(list_of_seeds, list_of_instances, is_SB):
    num_vehicles = 2
    # policy = policies.inngjerdingen_moeller.PILOT()
    policy = policies.hlv_master.BS_PILOT(criticality_weights_sets=[[0.3, 0.15, 0, 0.2, 0.1], [0.3, 0.5, 0, 0, 0.2], [0.6, 0.1, 0, 0.2, 0.05]], evaluation_weights=[0.85, 0, 0.05])
    for instance in list_of_instances:
        if instance == 'BG_W35':
            num_vehicles = 1
        filename=str(instance)+".csv"
        test_seeds_mp(list_of_seeds, policy, is_SB, filename, num_vehicles, instance=instance)

def test_seeds_mp(list_of_seeds, policy, is_SB, filename, num_vehicles= NUM_VEHICLES, duration= DURATION, instance=None): #change duration and number of vehicles HERE!
    # #------------PROCESS----------------
    seeds = list_of_seeds
    q = mp.Queue()
    processes = []
    returned_simulators = []
    for seed in seeds:
        process = mp.Process(target=run_simulation, args = (seed, policy, is_SB, duration, num_vehicles, returned_simulators, instance))
        processes.append(process)
        process.start()
    
    while len(returned_simulators) != len(seeds):
        pass
    
    for simulator_metrics in returned_simulators:
        policies.hlv_master.manage_results.write_sim_results_to_file(filename, simulator_metrics, duration, append=True)
        # if we run PILOT policy: 
        filename_time = "sol_time_"+filename
        policies.hlv_master.manage_results.write_sol_time_to_file(filename_time, simulator_metrics)
        policies.hlv_master.manage_results.visualize(filename, simulator_metrics.metrics, 'Failed events')
        policies.hlv_master.manage_results.visualize(filename, simulator_metrics.metrics, 'battery violation')
        policies.hlv_master.manage_results.visualize(filename, simulator_metrics.metrics, 'trips')
        policies.hlv_master.manage_results.visualize(filename, simulator_metrics.metrics, 'starvations, no bikes')
        policies.hlv_master.manage_results.visualize(filename, simulator_metrics.metrics, 'starvations, no battery')
        policies.hlv_master.manage_results.visualize(filename, simulator_metrics.metrics, 'starvation')

        policies.hlv_master.manage_results.write_parameters_to_file('parameters_' + filename, policy, num_vehicles, duration)
        # output.write_csv(simulator,'./policies/hlv_master/simulation_results/different_policies/'+filename, hourly = False)
        # for branch in range(policy.number_of_successors):
        #     print(f"Branch {branch+1}: {simulator.metrics.get_aggregate_value('branch'+str(branch+1))}")
    policies.hlv_master.manage_results.visualize_aggregated_results(filename)



if __name__ == "__main__":

    duration = DURATION
    num_vehicles = NUM_VEHICLES
    
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
    
    test_policies(list_of_seeds=list_of_seeds, policy_dict=policy_dict, num_vehicles=num_vehicles, duration = duration)

    duration = time.time() - start_time
    print("Running time: ", str(duration))