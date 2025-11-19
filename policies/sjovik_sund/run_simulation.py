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

# Import visualization if needed
try:
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
 
import time
import multiprocessing as mp
import csv
 
 
def run_simulation(seed, policy, duration=24, num_vehicles=1, queue=None, INSTANCE=None):
  
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=duration)
   
    #INSTANCE = "NY_W31"
    INSTANCE = "TD_W34" 
    
     
    # Load initial state
    state = init_state.read_initial_state("instances/"+INSTANCE)
    state.set_seed(seed)
    vehicles = [policy for i in range(num_vehicles)]
    state.set_sb_vehicles(vehicles)  # this creates one vehicle for each policy in the list
    tstate = target_state.USTargetState()
    d = demand.Demand()
    simulator = sim.Simulator(
        initial_state=state,
        target_state=tstate,
        demand=d,
        start_time=START_TIME,
        duration=DURATION,
        verbose=True,
    )
    
    simulator.run()
    if queue is not None:
        queue.put(simulator)
    return simulator


 
def write_results_to_file(filename, simulator, duration, solve_time, seed, append=False):
    """
    Write simulation results to a CSV file.
    """
    results_dir = './policies/sjovik_sund/simulation_results/'
    os.makedirs(results_dir, exist_ok=True)
    filepath = results_dir + filename
   
    mode = 'a' if append else 'w'
    file_exists = os.path.isfile(filepath) and append
   
    with open(filepath, mode, newline='') as f:
        writer = csv.writer(f)
       
        # Write header if new file
        if not file_exists:
            writer.writerow([
                'Seed',
                'Duration (hours)',
                'Total Runtime (s)',
                'Failed Events',
                'Starvations',
                'Bike Starvations',
                'Long Congestions',
                'Short Congestions',
                'Total Trips',
                'Bike Departures',
                'Bike Arrivals',
                'Vehicle Arrivals',
                'Bike Deliveries',
                'Bike Pickups',
            ])
       
        # Write data row
        writer.writerow([
            seed,
            duration,
            round(solve_time, 2),
            simulator.state.metrics.get_aggregate_value('failed events'),
            simulator.state.metrics.get_aggregate_value('starvations'),
            simulator.state.metrics.get_aggregate_value('bike starvations'),
            simulator.state.metrics.get_aggregate_value('long congestions'),
            simulator.state.metrics.get_aggregate_value('short congestions'),
            simulator.state.metrics.get_aggregate_value('trips'),
            simulator.state.metrics.get_aggregate_value('bike departure'),
            simulator.state.metrics.get_aggregate_value('bike arrival'),
            simulator.state.metrics.get_aggregate_value('vehicle arrivals'),
            simulator.state.metrics.get_aggregate_value('num bike deliveries'),
            simulator.state.metrics.get_aggregate_value('num bike pickups'),
        ])
 
 
def write_parameters_to_file(filename, policy, num_vehicles, duration):
    """
    Write policy parameters to a text file.
    """
    results_dir = './policies/sjovik_sund/simulation_results/'
    os.makedirs(results_dir, exist_ok=True)
    filepath = results_dir + filename
   
    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SJOVIK SUND POLICY PARAMETERS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Policy Type: MILP-based (Direct Optimization)\n")
        f.write(f"Number of Vehicles: {num_vehicles}\n")
        f.write(f"Simulation Duration: {duration} hours\n\n")
       
        f.write("MILP Parameters:\n")
        f.write(f"  Time Horizon (T): {policy.time_horizon} periods\n")
        f.write(f"  Period Length (tau): {policy.tau} minutes\n")
        f.write(f"  Roaming Enabled: {policy.roaming}\n")
       
        if policy.weights:
            f.write(f"\nObjective Weights:\n")
            f.write(f"  Starvation Weight (w_S): {policy.weights[0]}\n")
            f.write(f"  Congestion Weight (w_C): {policy.weights[1]}\n")
            f.write(f"  Deviation Weight (w_D): {policy.weights[2]}\n")
            f.write(f"  Maintenance reward (r_M): {policy.weights[3]}\n")
        else:
            f.write(f"\nObjective Weights: Default (0.45, 0.45, 0.1)\n")


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
            print(f"Seed {list_of_seeds[i]}: Completed in {solve_time:.2f}s")
    
    else:
        # Run simulations sequentially (easier for debugging)
        for i, seed in enumerate(list_of_seeds):
            print(f"\nRunning seed {seed}...")
            start_solve = time.time()
            simulator = run_simulation(seed, policy, duration, num_vehicles)
            solve_time = time.time() - start_solve
            write_results_to_file(results_file, simulator, duration, solve_time, seed, append=(i > 0))
            print(f"Seed {seed}: Completed in {solve_time:.2f}s")
    
    print(f"\nResults written to: policies/sjovik_sund/simulation_results/{results_file}")

 
def test_policies(list_of_seeds, policy_dict, num_vehicles=1, duration=24*5, use_multiprocessing=True):
    """
    Test multiple policies with multiple seeds each.
    
    Args:
        list_of_seeds: List of random seeds to test for each policy
        policy_dict: Dictionary mapping policy names to policy instances
        num_vehicles: Number of vehicles to use
        duration: Simulation duration in hours
        use_multiprocessing: If True, run seeds in parallel; if False, run sequentially
    """
    for policy_name, policy in policy_dict.items():
        print(f"\n{'='*80}")
        print(f"Testing Policy: {policy_name}")
        print(f"{'='*80}\n")
        
        # Write policy parameters to file
        write_parameters_to_file(f'{policy_name}_parameters.txt', policy, num_vehicles, duration)
        
        # Test this policy with all seeds
        results_file = f'{policy_name}_results.csv'
        test_seeds(list_of_seeds, policy, results_file, num_vehicles, duration, use_multiprocessing)

 
if __name__ == "__main__":
   
    # Simulation settings
    duration = 24  # hours - full day simulation
    num_vehicles = 1  # Need at least 1 vehicle to test the policy!
   
   
    # Dictionary of policies to test
    policy_dict = {
        'sjovik_sund_policy': policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(roaming=False, time_horizon=6, tau=5, weights=[0.45,0.45,0.1,0.01])
        # Add more policy variations here
    }
   
    # Test parameters
    #list_of_time_horizons = [10, 15, 20, 25, 30]
    #list_of_tau = [3, 5, 10, 15]
   
    # Weight combinations: [w_S, w_C, w_D]
    weights_dict = {
        'balanced': [0.45, 0.45, 0.1, 0.01],
        'starvation_focus': [0.7, 0.2, 0.1, 0.01],
        'congestion_focus': [0.2, 0.7, 0.1, 0.01],
        'deviation_focus': [0.3, 0.3, 0.4, 0.01],
    }
   
    # List of seeds to test
    list_of_seeds = [1]  # Start with just 1 seed for debugging
   
    # Instances to test
    #list_of_instances = ['instances/BO_W31', 'instances/TD_W34', 'instances/OS_W34']
    list_of_instances = ['instances/TD_W34']
   
    # Start timing
    start_time = time.time()
   
    # Test 1: Test default policy with multiple seeds (no multiprocessing for debugging)
    test_policies(list_of_seeds=list_of_seeds, policy_dict=policy_dict, num_vehicles=num_vehicles, duration=duration, use_multiprocessing=True)
   
    # End timing
    total_duration = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total running time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"{'='*80}\n")
 