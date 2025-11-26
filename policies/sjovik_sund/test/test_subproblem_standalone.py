# -*- coding: utf-8 -*-
"""
Test file for the Sjovik Sund subproblem.
Tests the MILP with real simulation data.
"""

import os
import sys
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "Sub_problem"))
sys.path.insert(0, str(current_dir.parent.parent))

import sjovik_sund_subproblem
import sim
import demand
from init_state.wrapper import read_initial_state
import target_state
from helpers import timeInMinutes
from settings import SB_INSTANCE_FILE


def test_single_subproblem(filename, start_day, start_hour, t_state, time_horizon, tau, duration, number_of_vehicles, run_simulation=False):
    """
    Test a single subproblem with real simulation data.
    
    Args:
        run_simulation: If True, run simulation for 'duration' before solving MILP.
                       If False, solve MILP at initial state (t=0).
    """
    print(f"\nTesting with: {filename}")
    print(f"Start: Day {start_day}, Hour {start_hour}")
    print(f"Time horizon: {time_horizon} hours, tau: {tau} minutes")
    print(f"Run simulation first: {run_simulation}")
    
    # Initialize state
    test_state = read_initial_state(filename)
    test_state.set_seed(1)
    
    # Setup demand
    test_demand = demand.Demand()
    test_demand.update_demands(test_state, start_day, start_hour)
    
    # Setup target state
    start_time = timeInMinutes(hours=start_hour)
    t_state.update_target_state(test_state, start_day, start_hour)
    
    # Create simulator
    test_simul = sim.Simulator(
        initial_state=test_state,
        target_state=t_state,
        demand=test_demand,
        start_time=start_time,
        duration=duration,
        verbose=True,
    )
    
    # Optionally run simulation to get to a more interesting state
    if run_simulation:
        print(f"\nRunning simulation for {duration} minutes...")
        test_simul.run()
        print(f"Simulation complete. Current time: {test_state.time} minutes")
        
        # Print some statistics about the simulation run
        metrics = test_state.metrics
        print(f"  Failed events: {metrics.get_aggregate_value('failed events')}")
        print(f"  Total trips: {metrics.get_aggregate_value('trips')}")
        print(f"  Starvations: {metrics.get_aggregate_value('starvations')}")
    
    # Create data dictionary from current simulator state
    data = create_data_from_simulator(test_simul, test_state, time_horizon, tau, number_of_vehicles)
    
    # Run model
    print("\nRunning optimization...")
    m = sjovik_sund_subproblem.run_subproblem_model(data)
    
    # Print results
    if m.Status == 2:  # Optimal
        print(f"\n✓ Optimal solution found")
        print(f"  Runtime: {round(m.Runtime, 2)}s")
        print(f"  Objective: {round(m.ObjVal, 2)}")
        print(f"  MIP Gap: 0.0% (optimal)")
    elif m.Status == 9:  # Time limit
        print(f"\n⚠ Time limit reached")
        print(f"  Runtime: {round(m.Runtime, 2)}s")
        print(f"  Best objective: {round(m.ObjVal, 2) if m.SolCount > 0 else 'No solution'}")
        if m.SolCount > 0:
            print(f"  MIP Gap: {round(m.MIPGap * 100, 2)}%")
    else:
        print(f"\n✗ Model status: {m.Status}")
    
    return m, data


def create_data_from_simulator(simulator, state, time_horizon, tau, number_of_vehicles):
    """Convert simulator to MILP data format"""
    from policies.sjovik_sund.Sub_problem.subproblem_parameters import MILP_parameters
    
    data = MILP_parameters(simulator, time_horizon, weights=None, tau=tau)
    data.initalize_parameters()
    
    return data.to_dict()


def test_multiple_runs(filename, start_day, start_hour, t_state, time_horizon, tau, duration, number_of_runs, number_of_vehicles, run_simulation=False):
    """
    Run multiple tests and report average performance.
    
    Args:
        run_simulation: If True, run simulation before solving MILP.
    """
    results = []
    
    for run in range(number_of_runs):
        print(f"\n{'='*80}")
        print(f"Test run {run + 1}/{number_of_runs}")
        print(f"{'='*80}")
        
        m, data = test_single_subproblem(filename, start_day, start_hour, t_state, 
                                         time_horizon, tau, duration, number_of_vehicles, run_simulation)
        
        if m.Status in [2, 9] and m.SolCount > 0:
            gap = 0.0 if m.Status == 2 else m.MIPGap
            results.append({
                'runtime': m.Runtime,
                'objective': m.ObjVal,
                'gap': gap,
                'status': m.Status
            })
    
    # Print summary
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Completed runs: {len(results)}/{number_of_runs}")
        print(f"Average runtime: {round(sum(r['runtime'] for r in results) / len(results), 2)}s")
        print(f"Average objective: {round(sum(r['objective'] for r in results) / len(results), 2)}")
        print(f"Average MIP gap: {round(sum(r['gap'] for r in results) / len(results), 4)}")


if __name__ == "__main__":
    # Test configuration
    filename = SB_INSTANCE_FILE  # Uses 'instances/TD_W34' from settings
    
    START_DAY = 0  # 0 = Monday
    START_HOUR = 8  # 8:00 AM
    DURATION = timeInMinutes(hours=1)
    
    time_horizon = 5  # hours
    tau = 5  # minutes per period
    number_of_vehicles = 1
    
    # Target state
    tstate = target_state.USTargetState()
    
    # Test at initial state (no simulation run)
    print("\n" + "="*80)
    print("TEST 1: Solving MILP at initial state (t=0)")
    print("="*80)
    test_single_subproblem(filename, START_DAY, START_HOUR, tstate, 
                          time_horizon, tau, DURATION, number_of_vehicles, run_simulation=False)
    
    # Test after running simulation for 1 hour
    print("\n" + "="*80)
    print("TEST 2: Solving MILP after 1 hour of simulation")
    print("="*80)
    test_single_subproblem(filename, START_DAY, START_HOUR, tstate, 
                          time_horizon, tau, DURATION, number_of_vehicles, run_simulation=True)
    
    # Uncomment to run multiple tests:
    # test_multiple_runs(filename, START_DAY, START_HOUR, tstate, 
    #                   time_horizon, tau, DURATION, 5, number_of_vehicles, run_simulation=False)
