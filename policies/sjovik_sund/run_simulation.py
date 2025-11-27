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
 
 
class LoggingSimulator(sim.Simulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_logged_day = -1
        self.last_starvations = 0
        self.last_congestions = 0

    def full_step(self):
        super().full_step()
        
        # Check for 23:00 logging
        # Time is in minutes. 23:00 is 23*60 = 1380 minutes into the day.
        current_time = self.state.time
        day = int(current_time // (24*60))
        minute_of_day = current_time % (24*60)
        
        # We want to log once per day, when we pass 23:00 (1380 minutes)
        if day > self.last_logged_day and minute_of_day >= 1380:
            self.log_daily_metrics(day)
            self.last_logged_day = day
            
    def log_daily_metrics(self, day):
        starvations = self.state.metrics.get_aggregate_value('starvations')
        congestions = self.state.metrics.get_aggregate_value('long congestions')
        
        daily_starvations = starvations - self.last_starvations
        daily_congestions = congestions - self.last_congestions
        
        print(f"\n{'='*40}")
        print(f"DAY {day} SUMMARY (23:00)")
        print(f"{'='*40}")
        print(f"Accumulated Starvations: {starvations} (+{daily_starvations} today)")
        print(f"Accumulated Congestions: {congestions} (+{daily_congestions} today)")
        print(f"{'='*40}\n")
        
        self.last_starvations = starvations
        self.last_congestions = congestions


def run_simulation(seed, policy, duration=24, num_vehicles=1, queue=None, INSTANCE=None):
  
    START_TIME = timeInMinutes(hours=7)  # 7 AM
    DURATION = timeInMinutes(hours=duration)
   
    INSTANCE = "TD_W34_old"
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
                'Maintenance Time (minutes)',
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
            simulator.state.metrics.get_aggregate_value('maintenance time'),
        ])
 
 
def write_simulation_summary(filename, simulator, duration, policy, seed):
    """
    Write simulation summary including parameters, objective function, and routes.
    """
    results_dir = './policies/sjovik_sund/simulation_results/'
    os.makedirs(results_dir, exist_ok=True)
    filepath = results_dir + filename

    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"SJOVIK SUND POLICY SIMULATION SUMMARY (Seed {seed})\n")
        f.write("="*80 + "\n\n")
        
        # 1. Parameters
        f.write("--- PARAMETERS ---\n")
        f.write(f"Policy Type: MILP-based (Direct Optimization)\n")
        f.write(f"Number of Vehicles: {num_vehicles}\n")
        f.write(f"Simulation Duration: {duration:.1f} hours\n")
       
        if hasattr(policy, 'time_horizon'):
            f.write(f"MILP Parameters:\n")
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
            f.write(f"\nObjective Weights: Default (0.45, 0.45, 0.09, 0.01)\n")
        
        # 2. Accumulated Objective Function
        f.write("\n--- ACCUMULATED OBJECTIVE FUNCTION ---\n")
        
        # Get aggregate metrics 
        # Regner bare med long congestions her
        starvations = simulator.state.metrics.get_aggregate_value('starvations')
        congestions_long = simulator.state.metrics.get_aggregate_value('long congestions') # + simulator.state.metrics.get_aggregate_value('short congestions')
        maintenance_time = simulator.state.metrics.get_aggregate_value('maintenance time')
        congestions_short = simulator.state.metrics.get_aggregate_value('short congestions')
        #deviations = simulator.state.metrics.get_aggregate_value('deviation')
        
        # Calculate objective
        w_S, w_C, w_D, r_M = policy.weights if policy.weights else (0.45, 0.45, 0.09, 0.01)
        
        obj_val = (w_S * starvations) + (w_C * congestions_long) - (r_M * maintenance_time)
        
        f.write(f"Total Objective Value: {obj_val:.4f}\n")
        f.write(f"Breakdown:\n")
        f.write(f"  Starvations: {starvations} (Contribution: {w_S * starvations:.4f})\n")
        f.write(f"  Congestions: {congestions_long} (Contribution: {w_C * congestions_long:.4f})\n")
        #f.write(f"  Deviations: {deviations} (Contribution: {w_D * deviations:.4f})\n")
        f.write(f"  Maintenance Time: {maintenance_time:.2f} (Contribution: {-r_M * maintenance_time:.4f})\n")
        f.write(f"  Short Congestions: {congestions_short} (Contribution: {0})\n")
        
        # 3. Routes
        f.write("\n--- ACTUAL VEHICLE ROUTES ---\n")
        
        # Extract routes from policy attached to vehicles in the simulator
        policy_with_routes = None
        for vehicle in simulator.state.vehicles.values():
            if hasattr(vehicle.policy, 'vehicle_routes'):
                policy_with_routes = vehicle.policy
                break
        
        if policy_with_routes and policy_with_routes.vehicle_routes:
            for vehicle_id, route in policy_with_routes.vehicle_routes.items():
                f.write(f"Vehicle {vehicle_id} Route:\n")
                if not route:
                    f.write("  No route recorded\n\n")
                    continue
                
                # Sort by time to ensure chronological order
                route.sort(key=lambda x: x[0])
                
                # Write route with time information
                for i, (time_val, station_id) in enumerate(route):
                    # Convert time to day/hour/minute format
                    day = int(time_val // (24*60))
                    hour = int((time_val % (24*60)) // 60)
                    minute = int(time_val % 60)
                    
                    if i == 0:
                        f.write(f"  Start: {station_id} at Day {day}, Hour {hour:02d}:{minute:02d} (t={time_val:.1f})\n")
                    else:
                        # Calculate travel time from previous station
                        prev_time = route[i-1][0]
                        travel_duration = time_val - prev_time
                        f.write(f"  Move {i}: {station_id} at Day {day}, Hour {hour:02d}:{minute:02d} (t={time_val:.1f}) [+{travel_duration:.1f} min]\n")
                
                # Summary statistics
                total_time = route[-1][0] - route[0][0] if len(route) > 1 else 0
                unique_stations = len(set(station for _, station in route))
                f.write(f"  Summary: {len(route)} stops, {unique_stations} unique stations, {total_time:.1f} min total\n\n")
        else:
             f.write("No route information available.\n")
       

    print(f"Simulation summary written to: {filepath}")


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
            
            # Write summary for this seed
            summary_filename = f"{filename.replace('.csv', '')}_summary_seed_{list_of_seeds[i]}.txt"
            write_simulation_summary(summary_filename, simulator, duration, policy, list_of_seeds[i])
            
            print(f"Seed {list_of_seeds[i]}: Completed in {solve_time:.2f}s")
    
    else:
        # Run simulations sequentially (easier for debugging)
        for i, seed in enumerate(list_of_seeds):
            print(f"\nRunning seed {seed}...")
            start_solve = time.time()
            simulator = run_simulation(seed, policy, duration, num_vehicles)
            solve_time = time.time() - start_solve
            write_results_to_file(results_file, simulator, duration, solve_time, seed, append=(i > 0))
            
            # Write summary for this seed
            summary_filename = f"{filename.replace('.csv', '')}_summary_seed_{seed}.txt"
            write_simulation_summary(summary_filename, simulator, duration, policy, seed)
            
            print(f"Seed {seed}: Completed in {solve_time:.2f}s")
    
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
    duration = 24*5 # hours - (24 * 5) for five days
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

    service_weights = [0.45,0.45,0.1]
    maintenance_reward = 1
    alpha = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.7, 1.0]
    #weights = [w*(1-alpha) for w in service_weights] + [maintenance_reward*alpha]
   

    policy_dict = {}
    for alpha in alpha:
        weights = [w*(1-alpha) for w in service_weights] + [maintenance_reward*alpha]
        policy_name = f'sjovik_sund_alpha_{alpha:.2f}'
        policy_dict[policy_name] = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(
            roaming=False, time_horizon=6, tau=5, weights=weights
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
