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
import target_state
import policies
import policies.inngjerdingen_moeller
import sim
import demand
from helpers import timeInMinutes
import time

import multiprocessing as mp



def run_simulation(seed, policy, queue=None):
    #change common parameters for the different simulations here:
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=3)
    INSTANCE = 'TD_W34_old'
    WEEK = 34
    ###############################################################
    
    state = init_state.read_initial_state("instances/"+INSTANCE);
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
    # policies.inngjerdingen_moeller.manage_results.write_sim_results_to_file(filename, simulator, DURATION, append=True)
    data = policies.inngjerdingen_moeller.manage_results.write_sim_results_to_list(simulator, DURATION)
    if queue != None:
        queue.put(data)
    return data

def test_policies(number_of_seeds, policy_dict):
    for policy in policy_dict:
        filename= "policy_"+str(policy)+".csv"
        for seed in range (1, number_of_seeds+1):
            run_simulation(seed, filename, policy_dict[policy])
        policies.inngjerdingen_moeller.visualize_aggregated_results_2(filename)

def test_timehorizons(number_of_seeds, list_of_timehorizons):
    for horizon in list_of_timehorizons:
        filename = "time_horizon_"+str(horizon)+".csv"
        policy=policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True, time_horizon=horizon)
        for seed in range (1, number_of_seeds+1):
            run_simulation(seed, filename, policy)
        policies.inngjerdingen_moeller.visualize_aggregated_results_2(filename)

def test_weights(number_of_seeds, weight_set):
    for set in weight_set:
        filename= "weight_set_"+str(set)+".csv"
        policy = policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True, time_horizon=25, weights=weight_set[set])
        for seed in range (1, number_of_seeds+1):
            run_simulation(seed, filename, policy)
        
        policies.inngjerdingen_moeller.visualize_aggregated_results_2(filename)

def test1(number_of_seeds):
    #------------Pool Apply----------------
    # filename = "multi.csv"
    # policy = policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True,time_horizon=15)
    # pool = mp.Pool(mp.cpu_count())
    # seeds = [i for i in range(number_of_seeds)]
    # data_lists = [pool.apply(run_simulation, args=(seed, policy)) for seed in seeds]
    # pool.close()
    # print(data_lists)

    #------------PROCESS----------------
    policy = policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True,time_horizon=10)
    seeds = [i for i in range(number_of_seeds)]
    q = mp.Queue()
    processes = []
    return_values = []
    for seed in seeds:
        process = mp.Process(target=run_simulation, args = (seed, policy, q))
        processes.append(process)
        process.start()
    for process in processes:
        ret = q.get()   #will block
        return_values.append(ret)
    for process in processes:
        process.join()
    print(return_values)


    #------------NON MULTIPROCESSING----------------
    # policy = policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True,time_horizon=10)
    # result_list = []
    # for i in range(number_of_seeds):
    #     result = run_simulation(i, policy)
    #     result_list.append(result)
    # print(result_list)
    # policies.inngjerdingen_moeller.visualize_aggregated_results_2(filename)



if __name__ == "__main__":
    print("Before time starts")
    start_time = time.time()
    # policy_dict = dict(random = policies.RandomActionPolicy(), greedy = policies.GreedyPolicy(), inngjerdingen_moeller = policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True,time_horizon=25))
    # list_of_timehorizons = [15, 20, 25, 30]
    # weight_dict = dict(a = [0.3, 0.3, 0.3], b=[0, 0, 0], c=[0, 0, 0], d=[0, 0, 0]) #[W_S, W_R, W_D]
    test1(5)
    print("Duration with multi: ", time.time()-start_time)