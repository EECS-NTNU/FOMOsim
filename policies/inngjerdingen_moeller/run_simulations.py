import os 
import sys
from pathlib import Path
 
path = Path(__file__).parents[2]        # The path seems to be correct either way, sys.path.insert makes the difference
os.chdir(path)
# print(os. getcwd())
sys.path.insert(0, '') #make sure the modules are found in the new working directory

import init_state
import target_state
import policies
import policies.inngjerdingen_moeller
import sim
import demand
from helpers import timeInMinutes


def run_simulation(seed, filename, policy):
    #change common parameters for the different simulations here:
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=4)
    INSTANCE = 'TD_W34'
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
    policies.inngjerdingen_moeller.manage_results.write_sim_results_to_file(filename, simulator, DURATION, append=True)    


def test_policies(number_of_seeds, policy_dict):
    for policy in policy_dict:
        filename= "policy_"+str(policy)+".csv"
        for seed in range (1, number_of_seeds+1):
            run_simulation(seed, filename, policy_dict[policy])
        policies.inngjerdingen_moeller.visualize_aggregated_results_2(filename)

def test_timehorizons(number_of_seeds, list_of_timehorizons):
    for horizon in list_of_timehorizons:
        filename = "time_horizon_"+str(horizon)+".csv"
        policy=policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True, time_horizon=horizon,tau=5, weights=None)
        for seed in range (1, number_of_seeds+1):
            run_simulation(seed, filename, policy)
        policies.inngjerdingen_moeller.visualize_aggregated_results_2(filename)

def test_weights(number_of_seeds, weight_set):
    for set in weight_set:
        filename= "weight_set_"+str(set)+".csv"
        policy = policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True, time_horizon=25, tau=5, weights=weight_set[set])
        for seed in range (1, number_of_seeds+1):
            run_simulation(seed, filename, policy)
        policies.inngjerdingen_moeller.visualize_aggregated_results_2(filename)

def test1(number_of_seeds):
    filename = "greedy_test.csv"
    policy = policies.GreedyPolicy()
    for seed in range (1, number_of_seeds+1):
            run_simulation(seed, filename, policy)
    policies.inngjerdingen_moeller.visualize_aggregated_results_2(filename)


if __name__ == "__main__":
    policy_dict = dict(greedy = policies.GreedyPolicy(), inngjerdingen_moeller = policies.inngjerdingen_moeller.InngjerdingenMoellerPolicy(roaming=True,time_horizon=25,tau=5, weights=None))
    list_of_timehorizons = [25, 30]
    weight_dict = dict(a = [0.45, 0.45, 0.1], b=[0.1, 0.1, 0.8], c=[0.35, 0.35, 0.3], d=[0.3, 0.3, 0.4]) #[W_S, W_R, W_D]
    
    test1(10)
    # test_weights(10,weight_dict)
    # test_timehorizons(10,list_of_timehorizons)
    # test_policies(10,policy_dict)
        