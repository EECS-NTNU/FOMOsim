# -*- coding: utf-8 -*-
 
 
######################################################
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
#from sim.Bike import Bike

# Get workspace root (2 levels up from this file)
WORKSPACE_ROOT = Path(__file__).parents[2]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, '')

# Force Python to not use cached bytecode - critical for cluster environments
sys.dont_write_bytecode = True
# Clear any existing __pycache__ to ensure fresh imports
import importlib
if hasattr(importlib, 'invalidate_caches'):
    importlib.invalidate_caches()
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
 
# python policies/sjovik_sund/run_simulation.py > policies/sjovik_sund/output/output.txt
# Import visualization if needed
'''
try:
    from policies.sjovik_sund.scripts.route_visualization.visualize_subproblem import Visualizer
    VISUALIZATION_AVAILABLE = True
except ModuleNotFoundError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"Warning: Visualization not available ({e})")
'''


MAINTENANCE_ENABLED = False
 

# Import logging utilities
from policies.sjovik_sund.simulation_logging import (
    LoggingSimulator,
    write_hourly_metrics_to_file,
    write_results_to_file,
    #write_simulation_summary,
    #write_vehicle_visits_to_file,
    #write_station_hourly_metrics_to_file,
    #write_bike_movements_to_file,
    #write_trip_requests_to_file,
    write_component_failures_to_file
)
from policies.sjovik_sund.operational_logging import OperationalLogger

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SimulationConfig:
    """Centralized configuration for simulation runs."""
    
    # === Time Settings ===
    start_hour: int = 7  # 7 AM start time
    
    # === Instance Settings ===
    default_instance: str = "TD_W34_old"
    available_instances: List[str] = field(default_factory=lambda: 
        ["TD_W34_old", "TD_W34_37", "OS_W31"])
    
    # === Instance-Specific Start Stations ===
    instance_start_stations: Dict[str, List[int]] = field(default_factory=lambda: {
        "TD": [0, 5, 10, 15, 20, 25, 30, 35, 40],
        "OS": [4, 5, 10, 15, 20, 25, 30, 35, 40],
        "EH": [0, 1, 2, 3, 4, 5, 6, 7] * 3 + [0, 1, 2]
    })
    
    # === Policy Weights ===
    service_weights: List[float] = field(default_factory=lambda: [0.45, 0.45, 0.1])
    maintenance_reward: float = 1.0
    
    # === Alpha Values ===
    default_alpha_with_maintenance: float = 0.25
    default_alpha_without_maintenance: float = 0.0
    
    # === MILP Policy Parameters ===
    tau: int = 5  # Time discretization in minutes
    default_time_horizon: int = 5  # Number of periods to look ahead
    policy_hour_from: int = 7  # Policy active from 7 AM
    policy_hour_to: int = 7  # Policy active until 11 PM
    roaming: bool = False
    
    # === Maintenance Settings ===
    default_maintenance_limit: float = 0.2
    
    # === CLI Defaults ===
    default_seed: int = 1
    default_nsims: int = 1
    default_vehicles: int = 1
    default_duration_hours: int = 24*365*2 # 5 days (3mnd)

    # === Operational Debug Logging ===
    operation_logging_enabled: bool = True
    operation_logging_include_bike_ids: bool = True
    
    # === Target State ===
    # Options: "half_capacity", "equal_prob", "us"
    target_state_type: str = "half_capacity"
    
    def get_start_stations(self, instance_name: str) -> List[int]:
        """Get start stations list based on instance name prefix."""
        for prefix, stations in self.instance_start_stations.items():
            if prefix in instance_name:
                return stations
        return [0]  # Fallback default
    
    def get_default_alpha(self, maintenance_enabled: bool) -> float:
        """Get default alpha value based on maintenance setting."""
        return (self.default_alpha_with_maintenance 
                if maintenance_enabled 
                else self.default_alpha_without_maintenance)
    
    def get_target_state_instance(self):
        """Return the appropriate target state instance."""
        if self.target_state_type == "half_capacity":
            return target_state.HalfCapacityTargetState()
        elif self.target_state_type == "equal_prob":
            return target_state.EqualProbTargetState()
        elif self.target_state_type == "us":
            return target_state.USTargetState()
        else:
            return target_state.HalfCapacityTargetState()  # Default
    
    def calculate_weights(self, alpha: float) -> List[float]:
        """Calculate final weights from alpha and service weights."""
        return ([w * (1 - alpha) for w in self.service_weights] + 
                [self.maintenance_reward * alpha])


def run_simulation(seed, policy, duration=24, num_vehicles=1, queue=None, instance_name=None, config=None):
    """Run a single simulation with given parameters.
    
    Args:
        seed: Random seed
        policy: Policy instance to use
        duration: Simulation duration in hours
        num_vehicles: Number of vehicles
        queue: Multiprocessing queue for results (optional)
        instance_name: Instance name (uses config default if None)
        config: SimulationConfig instance (creates default if None)
    """
    if config is None:
        config = SimulationConfig()
    
    START_TIME = timeInMinutes(hours=config.start_hour)
    DURATION = timeInMinutes(hours=duration)
 
    # Use config default if not provided
    if instance_name is None:
        instance_name = config.default_instance
    
    INSTANCE = instance_name

    # Load initial state using workspace-relative path
    instance_path = WORKSPACE_ROOT / "instances" / INSTANCE
    state = init_state.read_initial_state(str(instance_path))
    state.set_seed(seed)

    # Initialize bike maintenance criticality AFTER setting seed for deterministic results
    if MAINTENANCE_ENABLED:
        state.initialize_bike_maintenance()

    # In the initialization section:
    """if ENABLE_COMPONENT_FAILURES:
        all_bikes = state.get_all_bikes()
        print(f"\nInitializing component failure tracking for {len(all_bikes)} bikes...")
        
        print("Component failure tracking initialized.\n")"""

    vehicles = [policy for i in range(num_vehicles)]
    state.set_sb_vehicles(vehicles)  # this creates one vehicle for each policy in the list
 
    # Use config for target state
    tstate = config.get_target_state_instance()
    start_stations = config.get_start_stations(INSTANCE)
 
    # Distribute vehicles to start stations
    for i in range(num_vehicles):
        vehicle_id = f"V{i}"
        if vehicle_id in state.vehicles:
            station_id = f"S{start_stations[i % len(start_stations)]}"
            if station_id in state.locations:
                state.vehicles[vehicle_id].location = state.locations[station_id]
 
 
    d = demand.Demand()

    # Optional step-by-step operational logger (vehicle movement + actions)
    operation_logger = OperationalLogger(
        enabled=config.operation_logging_enabled,
        include_bike_ids=config.operation_logging_include_bike_ids,
    )
    state.operation_logger = operation_logger

    simulator = LoggingSimulator(
        initial_state=state,
        target_state=tstate,
        demand=d,
        start_time=START_TIME,
        duration=DURATION,
        verbose=True,
    )
    simulator.operation_logger = operation_logger

    if operation_logger.enabled:
        print("[OPS] Operational logging enabled (step-by-step vehicle/action trace)")
 

    print(
        f"Running simulation with duration {duration}, vehicles {num_vehicles}, "
        f"seed {seed}, Instance {INSTANCE} and weights {policy.weights}"
    )

    if ENABLE_COMPONENT_FAILURES:
        print("Component failure simulation: ENABLED")
  
    policy.maintenance_enabled = MAINTENANCE_ENABLED
  
    simulator.run()
  
    if queue is not None:
        queue.put(simulator)
    return simulator


def write_simulation_outputs(simulator, filename, seed, policy, duration, num_vehicles, append_to_results=False):
    """Write all output files for a single simulation run.
    
    Args:
        simulator: The completed simulator instance
        filename: Base filename for results (e.g., 'policy_results.csv')
        seed: Random seed used for this simulation
        policy: Policy instance with weights
        duration: Simulation duration in hours
        num_vehicles: Number of vehicles used
        append_to_results: Whether to append to main results file (False for first seed)
    """
    # Determine solve time
    solve_time = simulator.state.time
    
    # Base filename without extension
    base_filename = filename.replace('.csv', '')
    
    # Extract alpha from policy weights if available
    alpha_value = policy.weights[3] if policy.weights and len(policy.weights) > 3 else None
    
    # Write main results file
    write_results_to_file(filename, simulator, duration, solve_time, seed, append=append_to_results)
    
    # Write hourly metrics for this seed
    hourly_filename = f"{base_filename}_hourly_seed_{seed}.csv"
    write_hourly_metrics_to_file(hourly_filename, simulator, seed)
    
    # Write vehicle visits for this seed
    visits_filename = f"{base_filename}_vehicle_visits_seed_{seed}.csv"
    #write_vehicle_visits_to_file(visits_filename, simulator, seed)
    
    # Write station hourly metrics for this seed
    station_hourly_filename = f"{base_filename}_station_hourly_seed_{seed}.csv"
    #write_station_hourly_metrics_to_file(station_hourly_filename, simulator, seed)
    
    # Write bike movements for this seed
    bike_movements_filename = f"{base_filename}_bike_movements_seed_{seed}.csv"
    #write_bike_movements_to_file(bike_movements_filename, simulator, seed, alpha_value)
    
    # Write trip requests for this seed
    trip_requests_filename = f"{base_filename}_trip_requests_seed_{seed}.csv"
    #write_trip_requests_to_file(trip_requests_filename, simulator, seed, alpha_value)
    
    # Write Vehicle Decisions (currently commented out)
    # decisions_filename = f"{base_filename}_vehicle_decisions_seed_{seed}.csv"
    # write_vehicle_decisions_to_file(decisions_filename, simulator, seed)

    # write component failures for this seed
    if ENABLE_COMPONENT_FAILURES:
        component_failures_filename = f"{base_filename}_component_failures_seed_{seed}.csv"
        write_component_failures_to_file(component_failures_filename, simulator, seed, alpha_value)

    # Write summary for this seed
    summary_filename = f"{base_filename}_summary_seed_{seed}.txt"
    #write_simulation_summary(summary_filename, simulator, duration, policy, seed, num_vehicles)
    
    # Print completion info
    print(f"Seed {seed}: Completed in {solve_time:.2f}s")

def test_seeds(list_of_seeds, policy, filename, num_vehicles=1, duration=24*5, use_multiprocessing=True, instance_name=None, config=None):
    """Test multiple seeds with the same policy.
    
    Args:
        list_of_seeds: List of random seeds to test
        policy: Policy instance to use
        filename: Base filename for results
        num_vehicles: Number of vehicles
        duration: Simulation duration in hours
        use_multiprocessing: Whether to run in parallel
        instance_name: Instance name
        config: SimulationConfig instance
    """
    if config is None:
        config = SimulationConfig()
    
    results_file = filename
  
    if use_multiprocessing:
        # Run simulations in parallel
        queue = mp.Queue()
        processes = []
  
        for seed in list_of_seeds:
            p = mp.Process(target=run_simulation, args=(seed, policy, duration, num_vehicles, queue, instance_name, config))
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
            write_simulation_outputs(
                simulator=simulator,
                filename=results_file,
                seed=list_of_seeds[i],
                policy=policy,
                duration=duration,
                num_vehicles=num_vehicles,
                append_to_results=(i > 0)
            )
  
    else:
        # Run simulations sequentially (easier for debugging)
        for i, seed in enumerate(list_of_seeds):
            print(f"\nRunning seed {seed}...")
            start_solve = time.time()
            simulator = run_simulation(seed, policy, duration, num_vehicles, instance_name=instance_name, config=config)
            solve_time = time.time() - start_solve
            
            write_simulation_outputs(
                simulator=simulator,
                filename=results_file,
                seed=seed,
                policy=policy,
                duration=duration,
                num_vehicles=num_vehicles,
                append_to_results=(i > 0)
            )
  
    print(f"\nResults written to: policies/sjovik_sund/simulation_results/{results_file}")
 
 
def test_policies(list_of_seeds, policy_dict, num_vehicles=1, duration=24*5, use_multiprocessing=False, instance_name=None, config=None):
    """Test multiple policies with multiple seeds.
    
    Args:
        list_of_seeds: List of random seeds to test
        policy_dict: Dictionary of policy_name -> policy instance
        num_vehicles: Number of vehicles
        duration: Simulation duration in hours
        use_multiprocessing: Whether to run in parallel
        instance_name: Instance name
        config: SimulationConfig instance
    """
    if config is None:
        config = SimulationConfig()
    
    for policy_name, policy in policy_dict.items():
        print(f"\n{'='*80}")
        print(f"Testing Policy: {policy_name}")
        print(f"{'='*80}\n")
  
        # Test this policy with all seeds
        results_file = f'{policy_name}_results.csv'
        test_seeds(list_of_seeds, policy, results_file, num_vehicles, duration, use_multiprocessing, instance_name, config)
 
 
if __name__ == "__main__":
    # Create centralized configuration
    config = SimulationConfig()
    
    # ---- argument parsing for CLI ----
    parser = argparse.ArgumentParser(
        description="Run Sjovik & Sund simulation with different alpha values."
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        help=(
            "List of alpha values to test. "
            "If omitted, uses built-in defaults depending on MAINTENANCE_ENABLED."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.default_seed,
        help=f"Starting random seed (default: {config.default_seed}).",
    )
    parser.add_argument(
        "--nsims",
        type=int,
        default=config.default_nsims,
        help=f"Number of simulations (different seeds) per policy (default: {config.default_nsims}).",
    )
    parser.add_argument(
        "--time_horizon",
        type=int,
        default=config.default_time_horizon,
        help=f"Time horizon (T) for the MILP look-ahead policy (default: {config.default_time_horizon}).",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=config.default_instance,
        choices=config.available_instances,
        help=f"Instance to use for simulation (default: {config.default_instance}).",
    )
    parser.add_argument(
        "--vehicles",
        type=int,
        default=config.default_vehicles,
        help=f"Number of vehicles to use in the simulation (default: {config.default_vehicles}).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=config.default_duration_hours,
        help=f"Simulation duration in hours (default: {config.default_duration_hours} hours = 5 days).",
    )
    parser.add_argument(
        "--maintenance_limit",
        type=float,
        default=config.default_maintenance_limit,
        help=f"Maintenance criticality limit to check for maintenance actions (default: {config.default_maintenance_limit}).",
    )

    args = parser.parse_args()
 
    # Override global settings with command line arguments
    import settings
    settings.MAINTENANCE_LIMIT_TO_CHECK = args.maintenance_limit
    print(f"Using MAINTENANCE_LIMIT_TO_CHECK = {settings.MAINTENANCE_LIMIT_TO_CHECK}")
    
    # Simulation settings
    duration = args.duration
    num_vehicles = args.vehicles

    # Determine alpha values: from CLI if provided, otherwise defaults
    if args.alphas is not None:
        alpha_values = args.alphas
    else:
        alpha_values = [config.get_default_alpha(MAINTENANCE_ENABLED)]
 
    # Determine seeds: start at args.seed, run nsims seeds
    start_seed = args.seed
    list_of_seeds = list(range(start_seed, start_seed + args.nsims))

    timestamp = datetime.now().strftime("%m%d%H%M")

    policy_dict = {}
    for alpha in alpha_values:
        weights = config.calculate_weights(alpha)
        
        # Include instance, vehicles, duration, time horizon, timestamp, and seed in filename
        policy_name = (
            f"sjovik_sund_{args.instance}_V{num_vehicles}_D{duration}h_T{args.time_horizon}_"
            f"{timestamp}_seed{start_seed}_alpha{alpha:.3f}"
        )
        
        policy_dict[policy_name] = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(
            roaming=config.roaming,
            time_horizon=args.time_horizon,
            tau=config.tau,
            weights=weights,
            hour_from=config.policy_hour_from,
            hour_to=config.policy_hour_to,
        )
 
    # Start timing
    start_time = time.time()
 
    # Test policies (no multiprocessing for debugging)
    test_policies(
        list_of_seeds=list_of_seeds,
        policy_dict=policy_dict,
        num_vehicles=num_vehicles,
        duration=duration,
        use_multiprocessing=False,
        instance_name=args.instance,
        config=config,
    )
 
    # End timing
    total_duration = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total running time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"{'='*80}\n")