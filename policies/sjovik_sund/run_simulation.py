# -*- coding: utf-8 -*-

######################################################
import os
import sys
from pathlib import Path
 
path = Path(__file__).parents[2]       
os.chdir(path)
sys.path.insert(0, '') 
################################################################
 
import init_state
import target_state
import policies
import policies.sjovik_sund.sjovik_sund_policy
import sim
import demand
import output
from helpers import timeInMinutes
from settings import *

#python policies/sjovik_sund/run_simulation.py > policies/sjovik_sund/output/output.txt                          

# Import visualization if needed
try:
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
 
import time
import multiprocessing as mp

# Import logging utilities
from policies.sjovik_sund.simulation_logging import (
    LoggingSimulator, 
    write_hourly_metrics_to_file,
    write_results_to_file,
    write_simulation_summary
)


def run_simulation(seed, policy, duration=24, num_vehicles=1, queue=None, INSTANCE=None):
  
    START_TIME = timeInMinutes(hours=7)  # 7 AM
    DURATION = timeInMinutes(hours=duration)
   
    #INSTANCE = "TD_W34_old"
    INSTANCE = "TD_W34_testinstans"
    #INSTANCE = "trondheim"
    #INSTANCE = "NY_W31"
    #INSTANCE = "OS_W31"
    #INSTANCE = "EH_W31"
     
    # Load initial state
    state = init_state.read_initial_state("instances/"+INSTANCE)
    state.set_seed(seed)
    vehicles = [policy for i in range(num_vehicles)]
    state.set_sb_vehicles(vehicles)  # this creates one vehicle for each policy in the list

    #tstate = target_state.USTargetState()
    #tstate = target_state.EqualProbTargetState()
    tstate = target_state.HalfCapacityTargetState()

    # Assign vehicles to start stations based on instance
    start_stations = [0]
    if "TD" in INSTANCE: start_stations = [0,5,10,15,20,25,30,35,40]
    elif "OS" in INSTANCE: start_stations = [4,5,10,15,20,25,30,35,40]
    elif "EH" in INSTANCE: start_stations = [0,1,2,3,4,5,6,7,0, 1,2,3,4,5,6,7,0,1, 2,3,4,5,6,7,0,1,2]

    # Distribute vehicles to start stations
    for i in range(num_vehicles):
        vehicle_id = f"V{i}"
        if vehicle_id in state.vehicles:
            station_id = f"S{start_stations[i % len(start_stations)]}"
            if station_id in state.locations:
                state.vehicles[vehicle_id].location = state.locations[station_id]
                print(f"Vehicle {vehicle_id} assigned to {station_id}")
    

    d = demand.Demand()
    simulator = LoggingSimulator(
        initial_state=state,
        target_state=tstate,
        demand=d,
        start_time=START_TIME,
        duration=DURATION,
        verbose=True,
    )
    
    print(f"Running simulation with duration {duration}, vehicles {num_vehicles}, seed {seed}, Instance {INSTANCE} and weights {policy.weights}")
    
    simulator.run()
    
    if queue is not None:
        queue.put(simulator)
    return simulator



def test_seeds(list_of_seeds, policy, filename, num_vehicles=1, duration=24*5, use_multiprocessing=True):
    """
    Test a single policy with multiple seeds and write results to CSV file.
    
    Args:
        list_of_seeds: List of random seeds to test
        policy: The policy instance to test
        filename: Name of the results CSV file (without path)
        num_vehicles: Number of vehicles to use
        duration: Simulation duration in hours
        use_multiprocessing: If True, run seeds in parallel; if False, run sequentially
    """
    results_file = filename
    
    if use_multiprocessing:
        # Run simulations in parallel
        queue = mp.Queue()
        processes = []
        
        for seed in list_of_seeds:
            p = mp.Process(target=run_simulation, args=(seed, policy, duration, num_vehicles, queue))
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete and collect results
        returned_simulators = []
        for process in processes:
            simulator = queue.get()  # Will block until result available
            returned_simulators.append(simulator)
        
        for process in processes:
            process.join()
        
        # Write all results
        for i, simulator in enumerate(returned_simulators):
            solve_time = simulator.state.time  # Total simulation time
            write_results_to_file(results_file, simulator, duration, solve_time, list_of_seeds[i], append=(i > 0))
            
            # Write hourly metrics for this seed (one file per run)
            hourly_filename = f"{filename.replace('.csv', '')}_hourly_seed_{list_of_seeds[i]}.csv"
            write_hourly_metrics_to_file(hourly_filename, simulator, list_of_seeds[i])
            
            # Write summary for this seed
            summary_filename = f"{filename.replace('.csv', '')}_summary_seed_{list_of_seeds[i]}.txt"
            write_simulation_summary(summary_filename, simulator, duration, policy, list_of_seeds[i], num_vehicles)
            
            print(f"Seed {list_of_seeds[i]}: Completed in {solve_time:.2f}s")
            print(f"Hourly metrics written to: policies/sjovik_sund/simulation_results/{hourly_filename}")
    
    else:
        # Run simulations sequentially (easier for debugging)
        for i, seed in enumerate(list_of_seeds):
            print(f"\nRunning seed {seed}...")
            start_solve = time.time()
            simulator = run_simulation(seed, policy, duration, num_vehicles)
            solve_time = time.time() - start_solve
            write_results_to_file(results_file, simulator, duration, solve_time, seed, append=(i > 0))
            
            # Write hourly metrics for this seed (one file per run)
            hourly_filename = f"{filename.replace('.csv', '')}_hourly_seed_{seed}.csv"
            write_hourly_metrics_to_file(hourly_filename, simulator, seed)
            
            # Write summary for this seed
            summary_filename = f"{filename.replace('.csv', '')}_summary_seed_{seed}.txt"
            write_simulation_summary(summary_filename, simulator, duration, policy, seed, num_vehicles)
            
            print(f"Seed {seed}: Completed in {solve_time:.2f}s")
            print(f"Hourly metrics written to: policies/sjovik_sund/simulation_results/{hourly_filename}")
    
    print(f"\nResults written to: policies/sjovik_sund/simulation_results/{results_file}")

 
def test_policies(list_of_seeds, policy_dict, num_vehicles=1, duration=24*5, use_multiprocessing=True):
    for policy_name, policy in policy_dict.items():
        print(f"\n{'='*80}")
        print(f"Testing Policy: {policy_name}")
        print(f"{'='*80}\n")
        
        # Test this policy with all seeds
        results_file = f'{policy_name}_results.csv'
        test_seeds(list_of_seeds, policy, results_file, num_vehicles, duration, use_multiprocessing)

 
if __name__ == "__main__":
   
    # Simulation settings
    duration = 24 # hours - (24 * 5) for one week
    num_vehicles = 1 # Need at least 1 vehicle to test the policy! 
    
    
    # Test parameters
    #list_of_time_horizons = [5,7,10,12]
    #list_of_tau = [3, 5, 7, 10]
   
    # Weight combinations: [w_S, w_C, w_D. r_M]
    weights_dict = {
    'baseline_balanced':    [0.45, 0.45, 0.10, 0.0],
    'starvation_high':      [0.70, 0.20, 0.10, 0.0],
    'starvation_medium':    [0.60, 0.30, 0.10, 0.0],
    'congestion_high':      [0.20, 0.70, 0.10, 0.0],
    'congestion_medium':    [0.30, 0.60, 0.10, 0.0],
    'deviation_high':       [0.30, 0.30, 0.40, 0.0],
    'deviation_medium':     [0.35, 0.35, 0.30, 0.0],
    'starv_cong_balanced':  [0.475, 0.475, 0.05, 0.0],
    'starv_cong_60_30':     [0.60, 0.35, 0.05, 0.0],
    'starv_cong_30_60':     [0.35, 0.60, 0.05, 0.0],
    }

    #service_weights = [0.45, 0.45, 0.1]
    #maintenance_weight = 1.0
    #alpha = 0.001
    
    # Calculate combined weights: [(1-alpha)*Service, alpha*Maintenance]
    # Result structure: [w_S, w_C, w_D, r_M]
   # weights = [w * (1 - alpha) for w in service_weights] + [maintenance_weight * alpha]
 
    service_weights = [0.45,0.45,0.1]
    maintenance_reward = 1
    alpha = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    #weights = [w*(1-alpha) for w in service_weights] + [maintenance_reward*alpha]
   

    policy_dict = {}
    for alpha in alpha:
        weights = [w*(1-alpha) for w in service_weights] + [maintenance_reward*alpha]
        policy_name = f'sjovik_sund_alphas_run_EH_test_{alpha:.3f}'
        policy_dict[policy_name] = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(
            roaming=False, time_horizon=6, tau=5, weights=weights, hour_from=7, hour_to=20
    )
    # Dictionary of policies to test
    #policy_dict = {
        #'sjovik_sund_policy': policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(roaming=False, time_horizon=6, tau=5, weights=weights)
        # Add more policy variations here
    #}
   

    # List of seeds to test
    list_of_seeds = [1]  # Start with just 1 seed for debugging
   
    # Instances to test
    #list_of_instances = ['instances/BO_W31', 'instances/TD_W34', 'instances/OS_W34']
    list_of_instances = ['instances/TD_W34','instances/OS_W34']
   
    # Start timing
    start_time = time.time()
   
    # Test 1: Test default policy with multiple seeds (no multiprocessing for debugging)
    test_policies(list_of_seeds=list_of_seeds, policy_dict=policy_dict, num_vehicles=num_vehicles, duration=duration, use_multiprocessing=True)
   
    # End timing
    total_duration = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total running time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"{'='*80}\n")
