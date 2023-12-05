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

def run_simulation(seed, policy, duration= settings_duration, num_vehicles= settings_num_vehicles, queue=None, INSTANCE=None):

    #change common parameters for the different simulations here:
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=duration)

    INSTANCE = SETTINGS_INSTANCE

    # INSTANCE = 'BG_W8'
    # INSTANCE = 'BG_W25'  # more demand
    # INSTANCE = 'BG_W35'
    # INSTANCE = 'BG_W45'
    # INSTANCE = 'BO_W31'
    # INSTANCE = 'EH_W10'
    # INSTANCE = 'EH_W22'
    # INSTANCE = 'EH_W31'
    # INSTANCE = 'EH_W50'
    # INSTANCE = 'NY_W31'
    # INSTANCE = 'OS_W10'
    # INSTANCE = 'OS_W22'
    # INSTANCE = 'OS_W31'
    # INSTANCE = 'OS_W34'  # more demand
    # INSTANCE = 'OS_W50'
    # INSTANCE = 'TD_W17'
    # INSTANCE = 'TD_W21'
    # INSTANCE = 'TD_W34_old'
    # INSTANCE = 'TD_W34'
    # INSTANCE = 'TD_W44'
    
    ###############################################################
    
    # state = init_state.read_initial_state("instances/ebike/"+INSTANCE)
    state = init_state.read_initial_state("instances/ebike_with_depot/"+INSTANCE)
    state.set_seed(seed)
    vehicles = [policy for i in range(num_vehicles)]
    state.set_vehicles(vehicles) # this creates one vehicle for each policy in the list
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
        policy=policies.hlm.BS_PILOT(time_horizon=horizon, max_depth=5, number_of_successors=7)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_criticality_weights(list_of_seeds, criticality_weights_dict):
    for set in criticality_weights_dict:
        filename= "crit_set_"+str(set)+".csv"
        policy=policies.hlm.BS_PILOT(criticality_weights_sets=criticality_weights_dict[set])
        test_seeds_mp(list_of_seeds, policy, filename)

def test_evaluation_weights(list_of_seeds, evaluation_weights_dict):
     for set in evaluation_weights_dict:
        filename= "evaluation_set_"+str(set)+".csv"
        policy=policies.hlm.BS_PILOT(evaluation_weights=evaluation_weights_dict[set])
        test_seeds_mp(list_of_seeds, policy, filename)

def test_discounting_factors(list_of_seeds, list_of_factors):
     for factor in list_of_factors:
        filename= "discounting_"+str(factor)+".csv"
        policy=policies.hlm.BS_PILOT(discounting_factor=factor)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_alpha_beta(list_of_seeds, alpha, beta_list):
     for beta in beta_list:
        filename= "branching_a_"+str(alpha)+"_b_"+str(beta)+".csv"
        policy=policies.hlm.BS_PILOT(max_depth=alpha, number_of_successors=beta, time_horizon=60)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_number_of_scenarios(list_of_seeds, scenario_list):
     for number in scenario_list:
        filename= "num_scenarios_"+str(number)+".csv"
        policy=policies.hlm.BS_PILOT(number_of_scenarios=number, max_depth=2, number_of_successors=7)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_num_vehicles(list_of_seeds, vehicles_list):
     for v in vehicles_list:
        filename= "num_vehicles_"+str(v)+"V.csv"
        policy=policies.hlm.BS_PILOT(max_depth=2, number_of_successors=5, time_horizon=40)
        # policy = policies.GreedyPolicy()
        # policy= policies.inngjerdingen_moeller.GreedyPolicyNeighborhoodInteraction()
        test_seeds_mp(list_of_seeds, policy, filename, num_vehicles=v)

def test_instances(list_of_seeds, list_of_instances):
    num_vehicles = 2
    # policy = policies.inngjerdingen_moeller.PILOT()
    policy = policies.hlm.BS_PILOT(criticality_weights_sets=[[0.3, 0.15, 0, 0.2, 0.1], [0.3, 0.5, 0, 0, 0.2], [0.6, 0.1, 0, 0.2, 0.05]], evaluation_weights=[0.85, 0, 0.05])
    for instance in list_of_instances:
        if instance == 'BG_W35':
            num_vehicles = 1
        filename=str(instance)+".csv"
        test_seeds_mp(list_of_seeds, policy, filename, num_vehicles, instance=instance)

def test_seeds_mp(list_of_seeds, policy, filename, num_vehicles= settings_num_vehicles, duration= settings_duration, instance=None): #change duration and number of vehicles HERE!
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
        policies.hlm.manage_results.write_sim_results_to_file(filename, simulator, duration, append=True)
        # if we run PILOT policy:
        filename_time = "sol_time_"+filename
        policies.hlm.manage_results.write_sol_time_to_file(filename_time, simulator)

        policies.hlm.manage_results.write_parameters_to_file('parameters_' + filename, policy, num_vehicles, duration)
        # output.write_csv(simulator,'./policies/hlm/simulation_results/different_policies/'+filename, hourly = False)
        # for branch in range(policy.number_of_successors):
        #     print(f"Branch {branch+1}: {simulator.metrics.get_aggregate_value('branch'+str(branch+1))}")
    policies.hlm.manage_results.visualize_aggregated_results(filename)



if __name__ == "__main__":
            
    # evaluation_weights = [0.4, 0.3, 0.3] #[avoided_viol, neighbor_roaming, improved deviation]
    # criticality_weights_sets=[[0.4, 0.1, 0.2, 0.2, 0.1], [0.2, 0.4, 0.2, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1, 0.4]] #[time_to_viol, dev_t_state, neigh_crit, dem_crit, driving_time] 
    # criticality_weights_sets = [[0.4, 0.1, 0.2, 0.2, 0.1]]
    duration = settings_duration
    num_vehicles = settings_num_vehicles
    
    # policy_dict = dict(greedy_neigh = policies.inngjerdingen_moeller.GreedyPolicyNeighborhoodInteraction())
    # policy_dict = dict(pilot_no_roaming = policies.inngjerdingen_moeller.PILOT(criticality_weights_sets=[[0.3, 0.15, 0, 0.2, 0.1], [0.3, 0.5, 0, 0, 0.2], [0.6, 0.1, 0, 0.2, 0.05]], evaluation_weights=[0.85, 0, 0.05]))
    # policy_dict = dict(pilot_roaming = policies.inngjerdingen_moeller.PILOT())
    # policy_dict = dict(Kloimüllner = policies.inngjerdingen_moeller.PILOT(1, 250))
    # policy_dict = dict(DoNothing = policies.do_nothing_policy.DoNothing(), Kloimüllner = policies.inngjerdingen_moeller.PILOT(1, 260), pilot_X_roaming = policies.inngjerdingen_moeller.PILOT(), FOMOgreedy = policies.GreedyPolicy(), greedy_neigh = policies.inngjerdingen_moeller.GreedyPolicyNeighborhoodInteraction())
    # policy_dict = dict(DoNothing = policies.do_nothing_policy.DoNothing(), pilot_X_roaming = policies.inngjerdingen_moeller.PILOT(), FOMOgreedy = policies.GreedyPolicy(), greedy_neigh = policies.inngjerdingen_moeller.GreedyPolicyNeighborhoodInteraction())
    # policy_dict = dict(Kloimüllner_5 = policies.inngjerdingen_moeller.PILOT(1, 5))
    # policy_dict = dict(greedy = policies.GreedyPolicy(), nothing=policies.do_nothing_policy.DoNothing())
    # policy_dict = dict(DoNothing = policies.do_nothing_policy.DoNothing())
    policy_dict = dict(pilot_roaming = policies.hlm.BS_PILOT(
        max_depth = settings_max_depth, 
        number_of_successors = settings_number_of_successors, 
        time_horizon = settings_time_horizon, 
        criticality_weights_set = settings_criticality_weights_sets, 
        evaluation_weights = settings_evaluation_weights, 
        number_of_scenarios = settings_number_of_scenarios, 
        discounting_factor = settings_discounting_factor
    ))
    
    # list_of_timehorizons = settings_list_of_timehorizons
    evaluation_weights = dict(k=[0.45, 0.1, 0.45], l=[0.33, 0.33, 0.33], m=[0.9, 0.05, 0.05])
    # criticality_weights = settings_criticality_weights
    # list_of_factors = settings_list_of_factors
    list_of_instances = settings_list_of_instances
  
    list_of_seeds = settings_list_of_seeds
  
    start_time = time.time()

    test_evaluation_weights(list_of_seeds=list_of_seeds, evaluation_weights_dict=evaluation_weights)
    # test_criticality_weights(list_of_seeds=list_of_seeds, criticality_weights_dict=criticality_weights)
    # test_policies(list_of_seeds=list_of_seeds, policy_dict=policy_dict, num_vehicles=num_vehicles, duration = duration)
    # test_instances(list_of_seeds, list_of_instances)
    # test_discounting_factors(list_of_seeds, list_of_factors)
    # test_alpha_beta(list_of_seeds, 2, [1,3,5,7,10])
    # test_number_of_scenarios(list_of_seeds, [0,1,10,100,500,1000,2000])
    # test_timehorizons(list_of_seeds, list_of_timehorizons)
    # test_num_vehicles(list_of_seeds,[1,2,3])

    duration = time.time() - start_time
    print("Running time: ", str(duration))

