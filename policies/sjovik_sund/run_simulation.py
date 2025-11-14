# -*- coding: utf-8 -*-
"""
Run simulations for Sjovik Sund MILP-based policy
No heuristics or PILOT - direct MILP optimization
"""
 
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
import policies.sjovik_sund.sjovik_sund_policy
import sim
import demand
import output
from helpers import timeInMinutes
from settings import *
 
import time
import multiprocessing as mp
import csv
 
 
def run_simulation(seed, policy, duration=DURATION, num_vehicles=NUM_VEHICLES, queue=None, instance=None):
    """
    Run a single simulation with the given seed and policy.
    """
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=duration)
 
    if instance:
        INSTANCE = instance
    else:
        INSTANCE = SB_INSTANCE_FILE
 
    print(f"\n{'='*80}")
    print(f"Running simulation with seed {seed}, instance: {INSTANCE}")
    print(f"{'='*80}\n")
   
    # Load initial state
    state = init_state.read_initial_state(INSTANCE)
    state.set_seed(seed)
   
    # Set up vehicles with policy
    print(f"Creating {num_vehicles} vehicles with policy: {policy.__class__.__name__}")
   
    # Workaround: set_sb_vehicles has a bug where it checks "if num_vehicles > 0"
    # Instead, manually create vehicles
    for i in range(num_vehicles):
        vehicle_id = "V" + str(len(state.vehicles))
        # Find a valid starting location (first station or depot)
        start_location = None
        if "S0" in state.locations:
            start_location = state.locations["S0"]
        elif "D0" in state.locations:
            start_location = state.locations["D0"]
        else:
            # Use first available station
            stations = list(state.stations.values())
            if stations:
                start_location = stations[0]
       
        if start_location:
            vehicle = sim.Vehicle(
                vehicle_id=vehicle_id,
                start_location=start_location,
                policy=policy,
                battery_inventory_capacity=VEHICLE_BATTERY_INVENTORY,
                bike_inventory_capacity=VEHICLE_BIKE_INVENTORY,
                is_station_based=True
            )
            state.vehicles[vehicle_id] = vehicle
            print(f"  Created vehicle {vehicle_id} at location {start_location.id}")
        else:
            print(f"  ERROR: No valid starting location found!")
   
    print(f"Vehicles created: {len(state.vehicles)} total")
   
    # Set up target state
    tstate = target_state.USTargetState()
    # Calculate day and hour from START_TIME
    day = (START_TIME // (24 * 60)) % 7  # Day of week (0-6)
    hour = (START_TIME // 60) % 24  # Hour of day (0-23)
    tstate.update_target_state(state, day, hour)
   
    # Set up demand
    dmand = demand.Demand()
   
    # Create and run simulator
    simulator = sim.Simulator(
        initial_state=state,
        target_state=tstate,
        demand=dmand,
        start_time=START_TIME,
        duration=DURATION,
        verbose=True,
    )
   
    start_solve = time.time()
    simulator.run()
    solve_time = time.time() - start_solve
   
    print(f"\n{'='*80}")
    print(f"Simulation completed for seed {seed}")
    print(f"Total simulation time: {solve_time:.2f} seconds")
    print(f"{'='*80}\n")
   
    if queue is not None:
        queue.put((simulator, solve_time))
   
    return simulator, solve_time
 
 
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
            if len(policy.weights) > 3:
                f.write(f"  Maintenance Reward (r_M): {policy.weights[3]}\n")
        else:
            f.write(f"\nObjective Weights: Default (0.45, 0.45, 0.1)\n")
 
 
def test_policies(list_of_seeds, policy_dict, num_vehicles, duration, use_multiprocessing=False):
    """
    Test multiple policies across multiple seeds.
    """
    for policy_name in policy_dict:
        print(f"\n{'#'*80}")
        print(f"Testing Policy: {policy_name}")
        print(f"Number of vehicles: {num_vehicles}")
        print(f"{'#'*80}\n")
       
        filename = str(policy_name) + ".csv"
        if use_multiprocessing:
            test_seeds_mp(list_of_seeds, policy_dict[policy_name], filename, num_vehicles, duration)
        else:
            test_seeds_single(list_of_seeds, policy_dict[policy_name], filename, num_vehicles, duration)
 
 
def test_time_horizons(list_of_seeds, list_of_time_horizons, num_vehicles=NUM_VEHICLES, duration=DURATION):
    """
    Test different time horizons for the MILP model.
    """
    for horizon in list_of_time_horizons:
        print(f"\n{'#'*80}")
        print(f"Testing Time Horizon: {horizon}")
        print(f"{'#'*80}\n")
       
        filename = f"time_horizon_{horizon}.csv"
        policy = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(time_horizon=horizon)
        test_seeds_mp(list_of_seeds, policy, filename, num_vehicles, duration)
 
 
def test_tau_values(list_of_seeds, list_of_tau, num_vehicles=NUM_VEHICLES, duration=DURATION):
    """
    Test different tau (period length) values for the MILP model.
    """
    for tau_val in list_of_tau:
        print(f"\n{'#'*80}")
        print(f"Testing Tau (Period Length): {tau_val} minutes")
        print(f"{'#'*80}\n")
       
        filename = f"tau_{tau_val}.csv"
        policy = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(tau=tau_val)
        test_seeds_mp(list_of_seeds, policy, filename, num_vehicles, duration)
 
 
def test_objective_weights(list_of_seeds, weights_dict, num_vehicles=NUM_VEHICLES, duration=DURATION):
    """
    Test different objective function weight combinations.
    weights format: [w_S, w_C, w_D] or [w_S, w_C, w_D, r_M]
    """
    for weight_name in weights_dict:
        print(f"\n{'#'*80}")
        print(f"Testing Weights: {weight_name} - {weights_dict[weight_name]}")
        print(f"{'#'*80}\n")
       
        filename = f"weights_{weight_name}.csv"
        policy = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(weights=weights_dict[weight_name])
        test_seeds_mp(list_of_seeds, policy, filename, num_vehicles, duration)
 
 
def test_num_vehicles(list_of_seeds, vehicles_list, duration=DURATION):
    """
    Test different numbers of vehicles.
    """
    for v in vehicles_list:
        print(f"\n{'#'*80}")
        print(f"Testing Number of Vehicles: {v}")
        print(f"{'#'*80}\n")
       
        filename = f"num_vehicles_{v}V.csv"
        policy = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy()
        test_seeds_mp(list_of_seeds, policy, filename, num_vehicles=v, duration=duration)
 
 
def test_instances(list_of_seeds, list_of_instances, num_vehicles=NUM_VEHICLES, duration=DURATION):
    """
    Test different instances/maps.
    """
    for instance in list_of_instances:
        print(f"\n{'#'*80}")
        print(f"Testing Instance: {instance}")
        print(f"{'#'*80}\n")
       
        filename = f"{instance}.csv"
        policy = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy()
        test_seeds_mp(list_of_seeds, policy, filename, num_vehicles, duration, instance=instance)
 
 
def test_seeds_single(list_of_seeds, policy, filename, num_vehicles=NUM_VEHICLES, duration=DURATION, instance=None):
    """
    Run simulations for multiple seeds sequentially (no multiprocessing) - for debugging.
    """
    seeds = list_of_seeds
   
    for seed in seeds:
        print(f"\n*** Starting simulation for seed {seed} ***")
        simulator, solve_time = run_simulation(seed, policy, duration, num_vehicles, queue=None, instance=instance)
       
        # Write results
        write_results_to_file(filename, simulator, duration, solve_time, seed, append=True)
       
        # Write detailed CSV output
        detailed_filename = f"detailed_{filename.replace('.csv', '')}_seed{seed}.csv"
        output.write_csv(simulator.state, f'./policies/sjovik_sund/simulation_results/{detailed_filename}', hourly=False)
   
    # Write parameters file once
    write_parameters_to_file('parameters_' + filename, policy, num_vehicles, duration)
   
    print(f"\n{'='*80}")
    print(f"All simulations completed for {filename}")
    print(f"Results saved to: ./policies/sjovik_sund/simulation_results/")
    print(f"{'='*80}\n")
 
 
def test_seeds_mp(list_of_seeds, policy, filename, num_vehicles=NUM_VEHICLES, duration=DURATION, instance=None):
    """
    Run simulations for multiple seeds using multiprocessing.
    """
    seeds = list_of_seeds
    q = mp.Queue()
    processes = []
    returned_data = []
 
    # Start all processes
    for seed in seeds:
        process = mp.Process(target=run_simulation, args=(seed, policy, duration, num_vehicles, q, instance))
        processes.append(process)
        process.start()
   
    # Collect results
    for process in processes:
        ret = q.get()  # will block
        returned_data.append(ret)
   
    # Wait for all processes to complete
    for process in processes:
        process.join()
   
    # Write results to files
    for i, (simulator, solve_time) in enumerate(returned_data):
        seed = seeds[i]
        write_results_to_file(filename, simulator, duration, solve_time, seed, append=True)
       
        # Write detailed CSV output
        detailed_filename = f"detailed_{filename.replace('.csv', '')}_seed{seed}.csv"
        output.write_csv(simulator.state, f'./policies/sjovik_sund/simulation_results/{detailed_filename}', hourly=False)
   
    # Write parameters file once
    write_parameters_to_file('parameters_' + filename, policy, num_vehicles, duration)
   
    print(f"\n{'='*80}")
    print(f"All simulations completed for {filename}")
    print(f"Results saved to: ./policies/sjovik_sund/simulation_results/")
    print(f"{'='*80}\n")
 
 
if __name__ == "__main__":
   
    # Simulation settings
    duration = 1  # hours
    num_vehicles = 1  # Need at least 1 vehicle to test the policy!
   
    # Default policy with standard parameters
    default_policy = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(
        time_horizon=25,
        tau=5,
        weights=None  # Uses default: [0.45, 0.45, 0.1]
    )
   
    # Dictionary of policies to test
    policy_dict = {
        'sjovik_sund_default': default_policy,
        # Add more policy variations here
    }
   
    # Test parameters
    list_of_time_horizons = [10, 15, 20, 25, 30]
    list_of_tau = [3, 5, 10, 15]
   
    # Weight combinations: [w_S, w_C, w_D, r_M]
    weights_dict = {
        'balanced': [0.45, 0.45, 0.1, 0.0],
        'starvation_focus': [0.7, 0.2, 0.1, 0.0],
        'congestion_focus': [0.2, 0.7, 0.1, 0.0],
        'deviation_focus': [0.3, 0.3, 0.4, 0.0],
        'with_maintenance': [0.4, 0.4, 0.1, 0.1],
        'maintenance_focus': [0.2, 0.2, 0.1, 0.5],
    }
   
    # List of seeds to test
    list_of_seeds = [10]  # Start with just 1 seed for debugging
   
    # Instances to test
    #list_of_instances = ['instances/BO_W31', 'instances/TD_W34', 'instances/OS_W34']
    list_of_instances = ['instances/TD_W34']
   
    # Start timing
    start_time = time.time()
   
    # Test 1: Test default policy with multiple seeds (no multiprocessing for debugging)
    test_policies(list_of_seeds=list_of_seeds, policy_dict=policy_dict, num_vehicles=num_vehicles, duration=duration, use_multiprocessing=False)
   
    # End timing
    total_duration = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total running time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"{'='*80}\n")
 