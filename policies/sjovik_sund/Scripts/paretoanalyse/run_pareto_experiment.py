"""
Pareto Front Experiment: Maintenance vs. Service Quality Tradeoff

This script runs multiple simulations with different maintenance reward weights (r_M)
to explore the Pareto frontier between maintenance investment and service quality.

Experiment Design:
- Vary r_M from 0.0 (no maintenance) to 0.5 (high maintenance priority)
- Keep other weights constant: w_S=0.45, w_C=0.45, w_D=0.1
- Run with multiple seeds for statistical significance
- Simulations start at hour 7 (consistent with other experiments)
- Track both service metrics (starvation, congestion, deviation) and maintenance metrics

Author: Generated for FOMOsim thesis computational study
Date: 2025-11-23
"""

import sys
import os
import time
import csv
from pathlib import Path

# Setup path (same as run_simulation.py)
path = Path(__file__).parents[2]
os.chdir(path)
sys.path.insert(0, '')

import init_state
import target_state
import demand
import policies.sjovik_sund.sjovik_sund_policy
import sim
from settings import *
from helpers import timeInMinutes


def collect_maintenance_metrics(state):
    """
    Collect detailed maintenance statistics from the simulation.
    
    Returns:
        dict: Dictionary of maintenance metrics
    """
    stations = state.get_stations()
    
    total_bikes = 0
    total_criticality = 0.0
    bikes_critical_05 = 0  # bikes with criticality > 0.5
    bikes_critical_07 = 0  # bikes with criticality > 0.7
    bikes_critical_09 = 0  # bikes with criticality > 0.9
    
    station_max_criticalities = []
    station_avg_criticalities = []
    
    for station in stations:
        bikes = list(station.bikes.values())
        for bike in bikes:
            if hasattr(bike, 'maintenance_criticality'):
                total_bikes += 1
                crit = bike.maintenance_criticality
                total_criticality += crit
                
                if crit > 0.5:
                    bikes_critical_05 += 1
                if crit > 0.7:
                    bikes_critical_07 += 1
                if crit > 0.9:
                    bikes_critical_09 += 1
        
        # Station-level stats
        if len(bikes) > 0:
            stats = station.get_maintenance_criticality_stats()
            station_avg_criticalities.append(stats['average'])
            station_max_criticalities.append(stats['max'])
    
    avg_criticality = total_criticality / total_bikes if total_bikes > 0 else 0.0
    avg_station_criticality = sum(station_avg_criticalities) / len(station_avg_criticalities) if station_avg_criticalities else 0.0
    max_station_criticality = max(station_max_criticalities) if station_max_criticalities else 0.0
    
    return {
        'total_bikes': total_bikes,
        'avg_criticality': avg_criticality,
        'avg_station_criticality': avg_station_criticality,
        'max_station_criticality': max_station_criticality,
        'bikes_critical_05': bikes_critical_05,
        'bikes_critical_07': bikes_critical_07,
        'bikes_critical_09': bikes_critical_09,
        'pct_critical_05': 100.0 * bikes_critical_05 / total_bikes if total_bikes > 0 else 0.0,
        'pct_critical_07': 100.0 * bikes_critical_07 / total_bikes if total_bikes > 0 else 0.0,
        'pct_critical_09': 100.0 * bikes_critical_09 / total_bikes if total_bikes > 0 else 0.0,
    }


def run_single_simulation(seed, weights, duration, num_vehicles, instance_path='instances/TD_W34'):
    """
    Run a single simulation with given parameters.
    
    Args:
        seed: Random seed
        weights: [w_S, w_C, w_D, r_M]
        duration: Simulation duration in hours
        num_vehicles: Number of vehicles
        instance_path: Path to instance data
        
    Returns:
        dict: Results dictionary with all metrics
    """
    print(f"\n{'='*80}")
    print(f"Running simulation: Seed={seed}, r_M={weights[3]:.3f}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Create policy
    policy = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(
        roaming=False,
        time_horizon=4,
        tau=5,
        weights=weights
    )
    
    # Initialize state
    state = init_state.read_initial_state(instance_path)
    state.set_seed(seed)
    vehicles = [policy for i in range(num_vehicles)]
    state.set_sb_vehicles(vehicles)
    
    # Create target state and demand
    tstate = target_state.EqualProbTargetState()
    d = demand.Demand()
    
    # Create simulator (start at hour 7, same as other simulations)
    START_TIME = timeInMinutes(hours=7)
    DURATION = timeInMinutes(hours=duration)
    
    simulator = sim.Simulator(
        duration=DURATION,
        initial_state=state,
        target_state=tstate,
        demand=d,
        start_time=START_TIME,
        verbose=False
    )
    
    # Run simulation
    simulator.run()
    
    solve_time = time.time() - start_time
    
    # Collect service metrics
    service_metrics = {
        'starvations': state.metrics.get_aggregate_value('starvations'),
        'bike_starvations': state.metrics.get_aggregate_value('bike starvations'),
        'long_congestions': state.metrics.get_aggregate_value('long congestions'),
        'short_congestions': state.metrics.get_aggregate_value('short congestions'),
        'failed_events': state.metrics.get_aggregate_value('failed events'),
        'total_trips': state.metrics.get_aggregate_value('trips'),
        'bike_departures': state.metrics.get_aggregate_value('bike departure'),
        'bike_arrivals': state.metrics.get_aggregate_value('bike arrival'),
        'vehicle_arrivals': state.metrics.get_aggregate_value('vehicle arrivals'),
        'bike_deliveries': state.metrics.get_aggregate_value('num bike deliveries'),
        'bike_pickups': state.metrics.get_aggregate_value('num bike pickups'),
    }
    
    # Collect maintenance metrics
    maintenance_metrics = collect_maintenance_metrics(state)
    
    # Combine all results
    results = {
        'seed': seed,
        'w_S': weights[0],
        'w_C': weights[1],
        'w_D': weights[2],
        'r_M': weights[3],
        'duration_hours': duration,
        'num_vehicles': num_vehicles,
        'solve_time_s': round(solve_time, 2),
        **service_metrics,
        **maintenance_metrics,
    }
    
    # Calculate aggregate metrics
    results['total_service_cost'] = (
        results['starvations'] + 
        results['long_congestions'] + 
        results['short_congestions']
    )
    results['service_level'] = 100.0 * (1.0 - results['total_service_cost'] / max(1, results['total_trips']))
    
    print(f"Completed in {solve_time:.2f}s")
    print(f"Service Cost: {results['total_service_cost']:.0f}, Avg Criticality: {results['avg_criticality']:.3f}")
    
    return results


def write_pareto_results(results_list, filename='pareto_experiment_results.csv'):
    """
    Write results to CSV file.
    
    Args:
        results_list: List of result dictionaries
        filename: Output filename
    """
    results_dir = Path('./policies/sjovik_sund/simulation_results/')
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / filename
    
    if not results_list:
        print("No results to write!")
        return
    
    # Get all keys from first result (assumes all have same keys)
    fieldnames = list(results_list[0].keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)
    
    print(f"\n{'='*80}")
    print(f"Results written to: {filepath}")
    print(f"Total experiments: {len(results_list)}")
    print(f"{'='*80}\n")


def run_pareto_experiment():
    """
    Main function to run the Pareto frontier experiment.
    """
    print("\n" + "="*80)
    print("PARETO FRONT EXPERIMENT: Maintenance vs. Service Quality")
    print("="*80 + "\n")
    
    # ============================================================================
    # EXPERIMENT CONFIGURATION
    # ============================================================================
    
    # Simulation settings
    duration = 24  # hours - Full day simulation
    num_vehicles = 1
    instance_path = 'instances/TD_W34'
    
    # Seeds for statistical robustness (start with 3-5, can increase to 10+ for final results)
    list_of_seeds = [1, 2, 3]
    
    # Pareto experiment: Vary r_M while keeping other weights constant
    # weights = [w_S, w_C, w_D, r_M]
    base_weights = [0.45, 0.45, 0.1]  # Service weights (constant)
    
    r_M_values = [
        0.0,    # Baseline: No maintenance consideration
        0.02,   # Very low priority
        0.05,   # Low priority
        0.1,    # Equal to deviation weight
        0.15,   # Medium priority
        0.2,    # High priority (2x deviation)
        0.3,    # Very high priority
        0.5,    # Extreme priority (5x deviation)
    ]
    
    print("Experiment Configuration:")
    print(f"  Duration: {duration} hours (starting at hour 7)")
    print(f"  Vehicles: {num_vehicles}")
    print(f"  Seeds: {list_of_seeds}")
    print(f"  Base weights: w_S={base_weights[0]}, w_C={base_weights[1]}, w_D={base_weights[2]}")
    print(f"  r_M values: {r_M_values}")
    print(f"  Total simulations: {len(list_of_seeds) * len(r_M_values)}")
    print()
    
    # ============================================================================
    # RUN EXPERIMENTS
    # ============================================================================
    
    all_results = []
    total_experiments = len(list_of_seeds) * len(r_M_values)
    experiment_count = 0
    
    start_time_total = time.time()
    
    for r_M in r_M_values:
        weights = base_weights + [r_M]
        
        for seed in list_of_seeds:
            experiment_count += 1
            print(f"\nExperiment {experiment_count}/{total_experiments}")
            
            try:
                results = run_single_simulation(
                    seed=seed,
                    weights=weights,
                    duration=duration,
                    num_vehicles=num_vehicles,
                    instance_path=instance_path
                )
                all_results.append(results)
                
            except Exception as e:
                print(f"ERROR in simulation (seed={seed}, r_M={r_M}): {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    
    write_pareto_results(all_results)
    
    total_time = time.time() - start_time_total
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Successful runs: {len(all_results)}/{total_experiments}")
    print(f"{'='*80}\n")
    
    # ============================================================================
    # QUICK SUMMARY
    # ============================================================================
    
    print("\nQuick Summary (averaged across seeds):")
    print(f"{'r_M':<8} {'Service Cost':<15} {'Avg Criticality':<18} {'% Critical (>0.7)':<20}")
    print("-" * 65)
    
    for r_M in r_M_values:
        # Get all results for this r_M
        r_M_results = [r for r in all_results if abs(r['r_M'] - r_M) < 0.001]
        
        if r_M_results:
            avg_service_cost = sum(r['total_service_cost'] for r in r_M_results) / len(r_M_results)
            avg_criticality = sum(r['avg_criticality'] for r in r_M_results) / len(r_M_results)
            avg_pct_critical = sum(r['pct_critical_07'] for r in r_M_results) / len(r_M_results)
            
            print(f"{r_M:<8.3f} {avg_service_cost:<15.1f} {avg_criticality:<18.3f} {avg_pct_critical:<20.1f}%")
    
    print("\nNext steps:")
    print("1. Run: python analyze_pareto_results.py")
    print("2. Check: ./policies/sjovik_sund/simulation_results/pareto_plots/")


if __name__ == "__main__":
    run_pareto_experiment()

