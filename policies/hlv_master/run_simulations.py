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

def run_simulation(seed, policy, duration= DURATION, num_vehicles= NUM_VEHICLES, queue=None, INSTANCE=None):

    #change common parameters for the different simulations here:
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=duration)

    INSTANCE = INSTANCE_FILE
    
    ###############################################################
    
    # state = init_state.read_initial_state("instances/ebike/"+INSTANCE)
    state = init_state.read_initial_state(sb_jsonFilename = "instances/"+INSTANCE, ff_jsonFilename="instances/Ryde/TR_random_100_matrixes")
    state.set_seed(seed)
    vehicles = [policy for i in range(num_vehicles)]
    state.set_sb_vehicles(vehicles)
    tstate = target_state.USTargetState()
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
    if queue != None:
        queue.put(simulator)
    return simulator


def test_policies(list_of_seeds, policy_dict, num_vehicles, duration):
    for policy in policy_dict:
        filename=str(policy)+".csv"
        test_seeds_mp(list_of_seeds, policy_dict[policy], filename, num_vehicles, duration)

def test_timehorizons(list_of_seeds, list_of_timehorizons):
    for horizon in list_of_timehorizons:
        filename = "time_horizon_"+str(horizon)+".csv"
        policy=policies.hlv_master.BS_PILOT(time_horizon=horizon, max_depth=5, number_of_successors=7)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_criticality_weights(list_of_seeds, criticality_weights_dict):
    for set in criticality_weights_dict:
        filename= "crit_set_"+str(set)+".csv"
        policy=policies.hlv_master.BS_PILOT(criticality_weights_set=criticality_weights_dict[set])
        test_seeds_mp(list_of_seeds, policy, filename)

def test_evaluation_weights(list_of_seeds, evaluation_weights_dict):
     for set in evaluation_weights_dict:
        filename= "evaluation_set_"+str(set)+".csv"
        policy=policies.hlv_master.BS_PILOT(evaluation_weights=evaluation_weights_dict[set])
        test_seeds_mp(list_of_seeds, policy, filename)

def test_discounting_factors(list_of_seeds, list_of_factors):
     for factor in list_of_factors:
        filename= "discounting_"+str(factor)+".csv"
        policy=policies.hlv_master.BS_PILOT(discounting_factor=factor)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_overflow_starvation(list_of_seeds, list_of_overflow, list_of_starvation):
     for factor in list_of_overflow:
        for factor2 in list_of_starvation:
            filename= "overflow_starvation_"+str(factor)+str(factor2)+".csv"
            policy=policies.hlv_master.BS_PILOT(overflow_criteria=factor, starvation_criteria = factor2)
            test_seeds_mp(list_of_seeds, policy, filename)

def test_alpha_beta(list_of_seeds, alpha, beta_list):
     for beta in beta_list:
        filename= "branching_a_"+str(alpha)+"_b_"+str(beta)+".csv"
        policy=policies.hlv_master.BS_PILOT(max_depth=alpha, number_of_successors=beta, time_horizon=60)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_number_of_scenarios(list_of_seeds, scenario_list):
     for number in scenario_list:
        filename= "num_scenarios_"+str(number)+".csv"
        policy=policies.hlv_master.BS_PILOT(number_of_scenarios=number, max_depth=2, number_of_successors=7)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_upper_threshold(list_of_seeds, upper_thresholds):
    for threshold in upper_thresholds:
        filename= "upper_threshold_"+str(threshold)+".csv"
        policy=policies.hlv_master.BS_PILOT(upper_thresholds=threshold)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_num_vehicles(list_of_seeds, vehicles_list):
     for v in vehicles_list:
        filename= "num_vehicles_"+str(v)+"V.csv"
        policy=policies.hlv_master.BS_PILOT(max_depth=2, number_of_successors=5, time_horizon=40)
        # policy = policies.GreedyPolicy()
        # policy= policies.inngjerdingen_moeller.GreedyPolicyNeighborhoodInteraction()
        test_seeds_mp(list_of_seeds, policy, filename, num_vehicles=v)

def test_instances(list_of_seeds, list_of_instances):
    num_vehicles = 2
    # policy = policies.inngjerdingen_moeller.PILOT()
    policy = policies.hlv_master.BS_PILOT(criticality_weights_sets=[[0.3, 0.15, 0, 0.2, 0.1], [0.3, 0.5, 0, 0, 0.2], [0.6, 0.1, 0, 0.2, 0.05]], evaluation_weights=[0.85, 0, 0.05])
    for instance in list_of_instances:
        if instance == 'BG_W35':
            num_vehicles = 1
        filename=str(instance)+".csv"
        test_seeds_mp(list_of_seeds, policy, filename, num_vehicles, instance=instance)

def test_seeds_mp(list_of_seeds, policy, filename, num_vehicles= NUM_VEHICLES, duration= DURATION, instance=None): #change duration and number of vehicles HERE!
    # #------------PROCESS----------------
    seeds = list_of_seeds
    q = mp.Queue()
    processes = []
    returned_simulators = []

    for seed in seeds:
        process = mp.Process(target=run_simulation, args = (seed, policy, duration, num_vehicles, q, instance))
        processes.append(process)
        process.start()
    for process in processes:
        ret = q.get()   #will block
        returned_simulators.append(ret)
    for process in processes:
        process.join()
    for simulator in returned_simulators:
        policies.hlv_master.manage_results.write_sim_results_to_file(filename, simulator, duration, append=True)
        # if we run PILOT policy:
        filename_time = "sol_time_"+filename
        policies.hlv_master.manage_results.write_sol_time_to_file(filename_time, simulator)
        policies.hlv_master.manage_results.visualize(filename, simulator.metrics, 'Failed events')
        policies.hlv_master.manage_results.visualize(filename, simulator.metrics, 'battery violation')
        policies.hlv_master.manage_results.visualize(filename, simulator.metrics, 'trips')
        policies.hlv_master.manage_results.visualize(filename, simulator.metrics, 'starvations, no bikes')
        policies.hlv_master.manage_results.visualize(filename, simulator.metrics, 'starvations, no battery')
        policies.hlv_master.manage_results.visualize(filename, simulator.metrics, 'starvation')

        policies.hlv_master.manage_results.write_parameters_to_file('parameters_' + filename, policy, num_vehicles, duration)
        # output.write_csv(simulator,'./policies/hlv_master/simulation_results/different_policies/'+filename, hourly = False)
        # for branch in range(policy.number_of_successors):
        #     print(f"Branch {branch+1}: {simulator.metrics.get_aggregate_value('branch'+str(branch+1))}")
    policies.hlv_master.manage_results.visualize_aggregated_results(filename)



if __name__ == "__main__":

    duration = DURATION
    num_vehicles = NUM_VEHICLES
    
    policy_dict = {
        'SB_base': policies.hlv_master.BS_PILOT(),
        # 'FF_base': policies.hlv_master.BS_PILOT_FF(),
        # 'SB_Collab2': policies.hlv_master.SB_Collab2(),
        # 'FF_Collab2': policies.hlv_master.FF_Collab2(),
        # 'Collab3': policies.hlv_master.Collab3(),
        # 'Collab4': policies.hlv_master.Collab4(),
                   }
    
    # list_of_instances = ['TD_34']
  
    list_of_seeds = SEEDS_LIST
  
    start_time = time.time()

    # test_evaluation_weights(list_of_seeds=list_of_seeds, evaluation_weights_dict=evaluation_weights)
    # test_upper_threshold(list_of_seeds=list_of_seeds, upper_thresholds=[50,60,70,80,90,100])
    # test_criticality_weights(list_of_seeds=list_of_seeds, criticality_weights_dict=criticality_weights)
    # test_overflow_starvation(list_of_seeds=list_of_seeds, list_of_overflow = overflow_criterias, list_of_starvation = starvation_criterias)
    test_policies(list_of_seeds=list_of_seeds, policy_dict=policy_dict, num_vehicles=num_vehicles, duration = duration)
    # test_instances(list_of_seeds, list_of_instances)
    # test_discounting_factors(list_of_seeds, list_of_factors)
    # test_alpha_beta(list_of_seeds, 2, [1,3,5,7,10])
    # test_number_of_scenarios(list_of_seeds, [0,1,10,100,500,1000,2000])
    # test_timehorizons(list_of_seeds, list_of_timehorizons)
    # test_num_vehicles(list_of_seeds,[1,2,3])

    duration = time.time() - start_time
    print("Running time: ", str(duration))

