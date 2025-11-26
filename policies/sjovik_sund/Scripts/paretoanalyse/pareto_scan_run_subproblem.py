#!/usr/bin/env python3
"""
Scan maintenance reward `r_M` by running the MILP subproblem standalone for a fixed
initial state (uses the same initial-state loader as `run_simulation.py`).

This script does NOT run the full simulator; it builds the `State` using
`init_state.read_initial_state(...)`, constructs `MILP_parameters`, and calls
`run_subproblem_model(data)` once per weight setting. Results are written to
`policies/sjovik_sund/output/pareto_scan_results.csv` and per-run JSON
breakdowns are written by the subproblem exporter.

Usage (from repo root with venv active):
    & ./.venv/Scripts\python.exe ./policies/sjovik_sund/Scripts/pareto_scan_run_subproblem.py

"""
import os
import sys
import time
import glob
import json

# Ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from policies.sjovik_sund.Sub_problem.subproblem_parameters import MILP_parameters
from policies.sjovik_sund.Sub_problem.sjovik_sund_subproblem import run_subproblem_model
import init_state
import target_state
import demand
import sim
import policies.sjovik_sund.sjovik_sund_policy
from helpers import timeInMinutes


def build_state_for_instance(instance_name='TD_W34'):
    """
    Build a realistic state by running a simulation for a few hours.
    This creates demand pressure and imbalances for the subproblem to handle.
    """
    # Load initial state
    state = init_state.read_initial_state(os.path.join('instances', instance_name))
    state.set_seed(42)
    
    # Create a vehicle with the policy (required for the subproblem!)
    policy = policies.sjovik_sund.sjovik_sund_policy.SjovikSundPolicy(
        roaming=False, time_horizon=6, tau=5, weights=[0.45, 0.45, 0.1, 0.01]
    )
    state.set_sb_vehicles([policy])
    
    # Create simulator and run for a few hours to build up demand
    tstate = target_state.USTargetState()
    d = demand.Demand()
    
    START_TIME = timeInMinutes(hours=7)  # Start at 7 AM
    WARMUP = timeInMinutes(hours=3)  # Run for 3 hours to 10 AM (busy time!)
    
    simulator = sim.Simulator(
        initial_state=state,
        target_state=tstate,
        demand=d,
        start_time=START_TIME,
        duration=WARMUP,
        verbose=False  # Quiet during warmup
    )
    
    print(f"  Running warmup simulation (7 AM → 10 AM) to create demand pressure...")
    simulator.run()
    print(f"  Warmup complete. State now at hour {simulator.state.hour()} with {len(simulator.state.vehicles)} vehicle(s)")
    
    # Return the warmed-up state with imbalances and active demand
    return simulator.state


def find_latest_breakdown():
    files = sorted(glob.glob('subproblem_objective_breakdown_*.json'))
    return files[-1] if files else None


def main():
    outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(outdir, exist_ok=True)
    csvfn = os.path.join(outdir, 'pareto_scan_results.csv')

    instance = 'TD_W34'
    print('Building state for instance', instance)
    state = build_state_for_instance(instance)

    # Baseline service weights (w_S, w_C, w_D) -- you can modify these
    base_wS, base_wC, base_wD = 0.45, 0.45, 0.10

    # Grid of maintenance reward values to scan
    rM_grid = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

    # Open results CSV and write header
    import csv
    with open(csvfn, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['r_M', 'model_obj', 'starv_contrib', 'cong_contrib', 'dev_contrib', 'maint_time', 'maint_contrib', 'breakdown_file'])

        for r in rM_grid:
            print('\nRunning subproblem with r_M =', r)
            weights = [base_wS, base_wC, base_wD, r]
            params = MILP_parameters(state, time_horizon=6, weights=weights, tau=5)
            params.initalize_parameters()
            data = params.to_dict()

            # Run subproblem (this will call the exporter and write a JSON file)
            model = run_subproblem_model(data)

            # Find latest breakdown JSON (best-effort)
            bf = find_latest_breakdown()
            starv = cong = dev = maint_time = maint_contrib = None
            if bf:
                try:
                    j = json.load(open(bf))
                    t = j.get('terms', {})
                    starv = t.get('starvation', {}).get('contribution')
                    cong = t.get('congestion', {}).get('contribution')
                    dev = t.get('deviation', {}).get('contribution')
                    maint_contrib = t.get('maintenance_time', {}).get('contribution')
                    maint_time = t.get('maintenance_time', {}).get('sum')
                except Exception:
                    pass

            writer.writerow([r, getattr(model, 'ObjVal', None), starv, cong, dev, maint_time, maint_contrib, bf])
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass

    print('\nWrote scan results to', csvfn)


if __name__ == '__main__':
    main()
