######################################################
import os 
import sys
from pathlib import Path
 
path = Path(__file__).parents[2]        # The path seems to be correct either way, sys.path.insert makes the difference
os.chdir(path) 
# print(os. getcwd())
sys.path.insert(0, '') #make sure the modules are found in the new working directory
################################################################

import init_state
# from init_state import read_initial_state
import target_state 
import policies
import sim
import demand
from helpers import timeInMinutes
import output

import time
import multiprocessing as mp

# import cProfile
# import pstats
# import io


def run_simulation(seed, policy, duration=24*5, queue=None):
    #change common parameters for the different simulations here:
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=duration)
    # INSTANCE = 'TD_W34_old'
    INSTANCE = 'OS_W31' 
    # INSTANCE = 'BG_W35'
    # INSTANCE = "NY_W31"
    ###############################################################
    
    state = init_state.read_initial_state("instances/"+INSTANCE)
    state.set_seed(seed)
    state.set_vehicles([policy, policy]) # this creates one vehicle for each policy in the list
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


def test_policies(list_of_seeds, policy_dict, duration=24*5):
    for policy in policy_dict:
        filename= "policy_"+str(policy)+".csv"
        test_seeds_mp(list_of_seeds, policy_dict[policy], filename, duration)


def test_timehorizons(list_of_seeds, list_of_timehorizons, duration=24*5):
    for horizon in list_of_timehorizons:
        filename = "time_horizon_"+str(horizon)+".csv"
        policy=policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True, time_horizon=horizon)
        test_seeds_mp(list_of_seeds, policy, filename, duration)


def test_weights(list_of_seeds, weight_set, duration=24*5):
    for set in weight_set:
        filename= "weight_set_"+str(set)+".csv"
        policy = policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True, time_horizon=25, weights=weight_set[set]) #HUSK Å ENDRE TH
        test_seeds_mp(list_of_seeds, policy, filename, duration)


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
        # output.visualize([simulator.metrics], metric="branch0")
        # output.visualize([simulator.metrics], metric="weight_set"+str([0.2, 0.2, 0.1, 0.1, 0.4]))
        for branch in range(policy.number_of_successors):
            print(f"Branch {branch}: {simulator.metrics.get_aggregate_value('branch'+str(branch))}")
        for weight_set in policy.crit_weights_sets:
            print(f"Weight set {weight_set}: {simulator.metrics.get_aggregate_value('weight_set'+str(weight_set))}")
    policies.inngjerdingen_moeller.manage_results.visualize_aggregated_results_2(filename)



if __name__ == "__main__":
            
    evaluation_weights = [0.4, 0.3, 0.3] #[avoided_viol, neighbor_roaming, improved deviation]
    # criticality_weights_sets=[[0.4, 0.1, 0.2, 0.2, 0.1], [0.2, 0.4, 0.2, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1, 0.4]] #[time_to_viol, dev_t_state, neigh_crit, dem_crit, driving_time] 
    criticality_weights_sets = [[0.4, 0.1, 0.2, 0.2, 0.1]]
    number_of_scenarios = 10
    
    # policy_dict = dict(pilot = policies.inngjerdingen_moeller.PILOT(2, 5, 30, criticality_weights_sets, evaluation_weights, number_of_scenarios), greedy = policies.GreedyPolicy(), greedy_neigh = policies.inngjerdingen_moeller.GreedyPolicyNeighborhoodInteraction())
    policy_dict = dict(pilot = policies.inngjerdingen_moeller.PILOT(3, 10, 40, criticality_weights_sets, evaluation_weights, number_of_scenarios))
    # policy_dict = dict(greedy = policies.GreedyPolicy([0.5,0.2,0.2,0.1]))
    
    #Kloimüllner:
    # policy_dict = dict(pilot = policies.inngjerdingen_moeller.PILOT(0, 250, 30, criticality_weights_sets, evaluation_weights, number_of_scenarios))
    
    # list_of_timehorizons = [25, 30]
    # weight_dict = dict(a = [0.45, 0.45, 0.1], b=[0.1, 0.1, 0.8], c=[0.35, 0.35, 0.3], d=[0.3, 0.3, 0.4]) #[W_S, W_R, W_D]
    
    # list_of_seeds_1=[0,1,2,3,4,5,6,7,8,9]
    list_of_seeds_1=[0,1,2,3,4]
    # list_of_seeds_1=[5,6,7,8,9] 
    # list_of_seeds_1=[1]

    # test_weights(list_of_seeds=list_of_seeds, weight_set=weight_dict, duration=24*5)
    # test_timehorizons(list_of_seeds=list_of_seeds_1, list_of_timehorizons=list_of_timehorizons, duration=24*5)
    start_time = time.time()
    # test_policies(list_of_seeds=list_of_seeds_1, policy_dict=policy_dict, duration=24*5)
    test_policies(list_of_seeds=list_of_seeds_1, policy_dict=policy_dict, duration=24*5)
    duration = time.time() - start_time
    print("Running time: ", str(duration))
    
    
    

# Profiling:    
    # pr = cProfile.Profile()
    # pr.enable()
    # test_policies(list_of_seeds=list_of_seeds_1, policy_dict=policy_dict, duration=24*5)
    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.print_stats()
    # with open("profiling.txt","w+") as f:
    #     f.write(s.getvalue())
     


 