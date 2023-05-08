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

import time
import multiprocessing as mp

# import cProfile
# import pstats


def run_simulation(seed, policy, duration=24*5, queue=None):
    #change common parameters for the different simulations here:
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=duration)
    
    # INSTANCE = 'TD_W34_old'
    # INSTANCE = 'OS_W31' 
    INSTANCE = 'BG_W35'
    # INSTANCE = "NY_W31"
    ###############################################################
    
    state = init_state.read_initial_state("instances/"+INSTANCE)
    state.set_seed(seed)
    state.set_vehicles([policy]) # this creates one vehicle for each policy in the list
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

    


def test_policies(list_of_seeds, policy_dict):
    for policy in policy_dict:
        filename= "policy_"+str(policy)+".csv"
        test_seeds_mp(list_of_seeds, policy_dict[policy], filename)

def test_timehorizons(list_of_seeds, list_of_timehorizons):
    for horizon in list_of_timehorizons:
        filename = "time_horizon_"+str(horizon)+".csv"
        policy=policies.inngjerdingen_moeller.PILOT(time_horizon=horizon)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_criticality_weights(list_of_seeds, criticality_weights_dict):
    for set in criticality_weights_dict:
        filename= "crit_set_BG_"+str(set)+".csv"
        policy=policies.inngjerdingen_moeller.PILOT(criticality_weights_sets=criticality_weights_dict[set])
        test_seeds_mp(list_of_seeds, policy, filename)

def test_evaluation_weights(list_of_seeds, evaluation_weights_dict):
     for set in evaluation_weights_dict:
        filename= "evaluation_set_TD_"+str(set)+".csv"
        policy=policies.inngjerdingen_moeller.PILOT(evaluation_weights=evaluation_weights_dict[set])
        test_seeds_mp(list_of_seeds, policy, filename)

def test_discounting_factors(list_of_seeds, list_of_factors):
     for factor in list_of_factors:
        filename= "discounting_BG_"+str(factor)+".csv"
        policy=policies.inngjerdingen_moeller.PILOT(discounting_factor=factor)
        test_seeds_mp(list_of_seeds, policy, filename)

def test_seeds_mp(list_of_seeds, policy, filename, duration=24*5):
    #------------PROCESS----------------
    seeds = list_of_seeds
    q = mp.Queue()
    processes = []
    returned_simulators = []
    for seed in seeds:
        process = mp.Process(target=run_simulation, args = (seed, policy, duration, q))
        processes.append(process)
        process.start()
    for process in processes:
        ret = q.get()   #will block
        returned_simulators.append(ret)
    for process in processes:
        process.join()
    for simulator in returned_simulators:
        policies.inngjerdingen_moeller.manage_results.write_sim_results_to_file(filename, simulator, duration, append=True)
        #if we run PILOT policy: 
        print(f"Accumulated solution time = {simulator.metrics.get_aggregate_value('accumulated solution time')}")
        print(f"Number of problems solved = {simulator.metrics.get_aggregate_value('number of problems solved')}")
        print(f"Average solution time =  {simulator.metrics.get_aggregate_value('accumulated solution time')/simulator.metrics.get_aggregate_value('number of problems solved')}")
        # output.visualize([simulator.metrics], metric="branch0")
        # output.visualize([simulator.metrics], metric="weight_set"+str([0.2, 0.2, 0.1, 0.1, 0.4]))
        # for branch in range(policy.number_of_successors):
        #     print(f"Branch {branch}: {simulator.metrics.get_aggregate_value('branch'+str(branch))}")
        # for weight_set in policy.crit_weights_sets:
        #     print(f"Weight set {weight_set}: {simulator.metrics.get_aggregate_value('weight_set'+str(weight_set))}")
    policies.inngjerdingen_moeller.manage_results.visualize_aggregated_results(filename)



if __name__ == "__main__":
            
    # evaluation_weights = [0.4, 0.3, 0.3] #[avoided_viol, neighbor_roaming, improved deviation]
    # criticality_weights_sets=[[0.4, 0.1, 0.2, 0.2, 0.1], [0.2, 0.4, 0.2, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1, 0.4]] #[time_to_viol, dev_t_state, neigh_crit, dem_crit, driving_time] 
    # criticality_weights_sets = [[0.4, 0.1, 0.2, 0.2, 0.1]]
    
    # policy_dict = dict(greedy = policies.GreedyPolicy(), greedy_neigh = policies.inngjerdingen_moeller.GreedyPolicyNeighborhoodInteraction())
    # policy_dict = dict(pilot = policies.inngjerdingen_moeller.PILOT())
    # policy_dict = dict(Kloimüllner = policies.inngjerdingen_moeller.PILOT(0, 250))
    # policy_dict = dict(nothing = policies.do_nothing_policy.DoNothing())
    # policy_dict = dict(greedy = policies.GreedyPolicy())
    
    # list_of_timehorizons = [20, 30, 40, 50, 60]
    # evaluation_weights = dict(a = [0.4, 0.3, 0.3], b=[0.8, 0.1, 0.1], c=[0.1, 0.8, 0.1], d=[0.1, 0.1, 0.8], e=[0.6, 0.1, 0.3], f=[0.3, 0.6, 0.1], g=[0.3, 0.1, 0.6], h=[0.6, 0.3, 0.1], i=[1.0, 0.0, 0.0], j=[0.45, 0.45, 0.1], k=[0.45, 0.1, 0.45], l=[0.33, 0.33, 0.33], m=[0.9, 0.05, 0.05], n=[0.95, 0.04, 0.01], o=[0.85, 0.1, 0.05], p=[0.9, 0.09, 0.01])
    criticality_weights = dict(a=[[0.2, 0.2, 0.2, 0.2, 0.2]], b=[[0.3, 0.15, 0.25, 0.2, 0.1]], c=[[0.2, 0.4, 0.2, 0.1, 0.1]], d=[[0.3, 0.3, 0.1, 0.1, 0.2]], e=[[0.2, 0.7, 0.05, 0.05, 0]], f=[[0.05, 0.9, 0.05, 0, 0]], g=[[0.1, 0.6, 0.1, 0.1, 0.1]], h=[[0.3, 0.5, 0, 0, 0.2]], i=[[0.9, 0, 0, 0.1, 0]], j=[[0.7, 0.05, 0.1, 0.1, 0.05]], k=[[0.6, 0.1, 0.05, 0.2, 0.05]], l=[[0.5, 0.05, 0.2, 0.05, 0.2]], m=[[1, 0, 0, 0, 0]], n=[[0, 1, 0, 0, 0]], o=[[0, 0, 1, 0, 0]], p=[[0, 0, 0, 1, 0]], q=[[1, 0, 0, 0, 0]])
    # list_of_factors = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    list_of_seeds=[0,1,2,3,4,5,6,7,8,9]
    # list_of_seeds=[0,1,2,3,4]
    # list_of_seeds=[5,6,7,8,9] 
    # list_of_seeds=[1]
    
    # profiler = cProfile.Profile()
    # profiler.enable()

    start_time = time.time()
    # test_evaluation_weights(list_of_seeds=list_of_seeds, evaluation_weights_dict=evaluation_weights)
    test_criticality_weights(list_of_seeds=list_of_seeds, criticality_weights_dict=criticality_weights)
    # test_policies(list_of_seeds=list_of_seeds, policy_dict=policy_dict)
    # test_discounting_factors(list_of_seeds, list_of_factors)
    duration = time.time() - start_time
    print("Running time: ", str(duration))

    
    # profiler.disable()
    # profiler.dump_stats('restats')  #add path
    # p = pstats.Stats('restats')
    # #p.sort_stats('time').print_stats(100) # print the stats after sorting by time
    # p.sort_stats('cumtime').print_stats(100) # print the stats after sorting by time



    
    
    


