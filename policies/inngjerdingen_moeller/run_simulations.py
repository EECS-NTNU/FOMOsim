from settings import *
import init_state
import init_state.json_source
import init_state.csv_source
import target_state
import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import policies.gleditsch_hagen
import policies.inngjerdingen_moeller
import sim
import output
import demand
from helpers import timeInMinutes

################# GLOBAL SETTINGS:  #####################
START_TIME = timeInMinutes(hours=7)
DURATION = timeInMinutes(hours=12)
INSTANCE = 'TD_W34_old'
WEEK = 34
#########################################################

def run_simulation(seed=1, filename, policy):
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

    results_visualizer = policies.inngjerdingen_moeller.manage_results.VisualizeResults(simulator)

    policies.inngjerdingen_moeller.manage_results.write_sim_results_to_file(filename, simulator, DURATION, append=True)    


def test_policies(number_of_seeds, policy_dict):
    for policy in policy_dict:
        filename= "policy_"+str(policy)
        for seed in range (1, number_of_seeds+1):
            run_simulation(seed,filename,policy_dict[policy])

def test_timehorizons(number_of_seeds, list_of_timehorizons):
    for horizon in list_of_timehorizons:
        filename = "time_horizon_"+str(horizon)
        policy = policies.inngjerdingen_moeller.inngjerdingen_moeller.InngjerdingenMoellerPolicy(time_horizon=horizon)
        for seed in range (1, number_of_seeds+1):
            run_simulation(seed,filename,policy_dict[policy])

def test_weights(number_of_seeds, weigth_sets):


######################Test parameters: ############################
policy_dict = dict(random = policies.RandomActionPolicy(), greedy = policies.GreedyPolicy(), inngjerdingen_moeller = policies.inngjerdingen_moeller.inngjerdingen_moeller.InngjerdingenMoellerPolicy(time_horizon=25))
list_of_timehorizons = [15, 20, 25, 30]
weight_dict = dict(a = [0.3, 0.3, 0.3], b=[0, 0, 0], c=[0, 0, 0], d=[0, 0, 0])
###################################################################


results_visualizer.visualize_aggregated_results('sim_test.csv')