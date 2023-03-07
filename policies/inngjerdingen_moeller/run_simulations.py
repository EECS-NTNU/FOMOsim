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

import multiprocessing as mp



def run_simulation(seed, policy, duration=24*5, queue=None):
    #change common parameters for the different simulations here:
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=duration)
    # INSTANCE = 'TD_W34_old'
    INSTANCE = 'OS_W31'
    # INSTANCE = 'BG_W35'
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
    policies.inngjerdingen_moeller.manage_results.visualize_aggregated_results_2(filename)



if __name__ == "__main__":
            
    evaluation_weights=[0.4, 0.33, 0.3] #[avoided_viol, neighbor_roaming, improved deviation]
    criticality_weights=[0.4, 0.1, 0.2, 0.2, 0.1] #[time_to_viol, dev_t_state, neigh_crit, dem_crit, driving_time] 
    policy_dict = dict(pilot = policies.inngjerdingen_moeller.PILOT(2, 10, 30, criticality_weights, evaluation_weights), greedy = policies.GreedyPolicy())
    
    # policy_dict = dict(milp_no_roaming = policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=False, time_horizon=20)) #for greedy_with_neighbors: crit_weights = [time_to_viol, dev_t_state, neigh_crit, dem_crit]
    # list_of_timehorizons = [25, 30]
    # weight_dict = dict(a = [0.45, 0.45, 0.1], b=[0.1, 0.1, 0.8], c=[0.35, 0.35, 0.3], d=[0.3, 0.3, 0.4]) #[W_S, W_R, W_D]
    
    list_of_seeds_1=[0,1,2,3,4,5,6,7,8,9] 
    # list_of_seeds_1=[1]

    # test_weights(list_of_seeds=list_of_seeds, weight_set=weight_dict, duration=24*5)
    # test_timehorizons(list_of_seeds=list_of_seeds_1, list_of_timehorizons=list_of_timehorizons, duration=24*5)
    test_policies(list_of_seeds=list_of_seeds_1, policy_dict=policy_dict, duration=24*5)
    