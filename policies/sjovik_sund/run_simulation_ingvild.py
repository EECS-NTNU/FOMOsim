# -*- coding: utf-8 -*-

# python policies/sjovik_sund/run_simulation_ingvild.py --vfa-model models/final_ablation_500ep_timefix/Imbalance_Squared_Temporal_alpha_0.1_20260429_202323/vfa_Imbalance_Squared_Temporal_seed1000.pkl --active-features rebalancing_imbalance squared_starvation_penalty squared_congestion_penalty gross_starvation_risk gross_congestion_risk --duration 672



# python policies/sjovik_sund/run_simulation_ingvild.py --vfa-model models/final_ablation_500ep_timefix/Imbalance_Squared_Temporal_alpha_0.1_20260429_202323/vfa_Imbalance_Squared_Temporal_seed1000.pkl --active-features rebalancing_imbalance squared_starvation_penalty squared_congestion_penalty gross_starvation_risk gross_congestion_risk --duration 672


######################################################
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
#from sim.Bike import Bike

# Get workspace root (2 levels up from this file)
WORKSPACE_ROOT = Path(__file__).parents[2]
STEADY_STATE_ODOMETER_DIR = (
    WORKSPACE_ROOT
    / "policies"
    / "sjovik_sund"
    / "simulation_results"
    / "steady_state_odometers"
)
AUTO_ODOMETER_STATS = "auto"
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
import policies.sjovik_sund.XPILOT_policy
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.vfa.HybridRolloutPolicy import HybridRolloutPolicy
from policies.sjovik_sund.mdp.reward import RewardCalculator, RewardConfig
from policies.do_nothing_policy import DoNothing
from policies.greedy_policy import GreedyPolicy
#from policies.greedy_policy_V2 import GreedyPolicyV2
from policies.greedy_policy_maintenance import GreedyMaintenancePolicy

import sim
import demand
import output
from helpers import timeInMinutes
from settings import *
import time
import multiprocessing as mp

# Maintenance is enabled iff component failures are enabled in settings
MAINTENANCE_ENABLED = ENABLE_COMPONENT_FAILURES
 

# Import logging utilities
from policies.sjovik_sund.simulation_logging import (
    LoggingSimulator,
    SimulationRunLogger,
    write_bike_movements_to_file,
    #write_hourly_metrics_to_file,
    write_results_to_file,
    write_daily_metrics_to_file,
    #write_simulation_summary,
    #write_vehicle_visits_to_file,
    #write_station_hourly_metrics_to_file,
    #write_bike_movements_to_file,
    #write_trip_requests_to_file,
    write_component_failures_to_file,
    #write_rl_decisions_to_file,
    #write_rl_decisions_to_file,
    write_vehicle_and_health_logs
)
from policies.sjovik_sund.operational_logging import OperationalLogger
from sim.bike_degradation_modeling.steady_state_odometer import (
    aggregate_seed_summaries,
    apply_component_odometer_initialization,
    collect_component_odometer_samples,
    iter_unique_bikes,
    summarize_component_odometers,
    write_csv_rows,
)

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SimulationConfig:
    """Centralized configuration for simulation runs."""
    
    # === Time Settings ===
    start_hour: int = 5  # 5 AM start time
    
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
    policy_hour_from: int = 6  # Policy active from 6 AM
    policy_hour_to: int = 20  # Policy active until 8 PM
    roaming: bool = False
    
    # === Maintenance Settings ===
    default_maintenance_limit: float = 0.2
    
    # === CLI Defaults ===
    default_seed: int = 42
    default_nsims: int = 1
    default_vehicles: int = 1
    default_duration_hours: int = 1344 # 56 days / 8 weeks

    # === Operational Debug Logging ===
    operation_logging_enabled: bool = False
    operation_logging_include_bike_ids: bool = True

    # === Initial Component Wear ===
    # If set, run_simulation samples component odometers from this CSV at startup.
    odometer_stats_path: str = AUTO_ODOMETER_STATS
    odometer_sampling_method: str = "triangular"
    odometer_sampling_bounds: str = "p05-p95"
    odometer_debug_bike_ids: List[str] = field(default_factory=lambda: ["B54"])
    
    # === Target State ===
    # Options: "half_capacity", "equal_prob", "us"
    target_state_type: str = "equal_prob"
    
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


def _resolve_odometer_stats_path(stats_path, instance_name: str) -> Path | None:
    """Resolve a concrete odometer stats CSV path.

    The default "auto" mode picks the newest aggregate CSV generated for the
    current instance under policies/sjovik_sund/simulation_results.
    """
    if stats_path is None:
        return None

    stats_path_str = str(stats_path)
    if stats_path_str.strip().lower() in {"", "none", "off", "false", "disabled"}:
        return None

    if stats_path_str.strip().lower() != AUTO_ODOMETER_STATS:
        return Path(stats_path_str)

    candidates = list(
        STEADY_STATE_ODOMETER_DIR.glob(
            f"{instance_name}_*/component_odometer_stats_aggregated.csv"
        )
    )
    if not candidates:
        candidates = list(
            STEADY_STATE_ODOMETER_DIR.glob("**/component_odometer_stats_aggregated.csv")
        )
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _log_debug_bike_odometer_initialization(
    state,
    seed,
    bike_ids: List[str],
    stats_path: Path,
    sampling_method: str,
    sampling_bounds: str,
) -> None:
    if not bike_ids:
        return

    wanted_ids = set(bike_ids)
    found_ids = set()
    for bike in iter_unique_bikes(state):
        bike_id = getattr(bike, "bike_id", None)
        if bike_id not in wanted_ids:
            continue

        found_ids.add(bike_id)
        component_odometers = getattr(bike, "component_odometers", {}) or {}
        '''print(
            f"[ODOMETER INIT] seed={seed} bike={bike_id}\n"
            f"  stats={stats_path}\n"
            f"  method={sampling_method} bounds={sampling_bounds}\n"
            f"  total_distance_km={getattr(bike, 'total_distance_km', 0.0):.6f}"
        )
        for component, odometer in sorted(component_odometers.items()):
            print(f"  {component}: {odometer:.6f}")'''

    missing_ids = wanted_ids - found_ids
    for bike_id in sorted(missing_ids):
        print(f"[ODOMETER INIT] seed={seed} bike={bike_id} not_found=true")


def run_simulation(
    seed,
    policy,
    duration=24,
    num_vehicles=2,
    queue=None,
    instance_name=None,
    config=None,
    run_logger=None,
    odometer_stats_path=None,
    odometer_sampling_method="triangular",
    odometer_sampling_bounds="p05-p95",
    verbose=True,
):
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
    state: sim.State = init_state.read_initial_state(str(instance_path))
    state.set_seed(seed)

    if odometer_stats_path is None:
        odometer_stats_path = config.odometer_stats_path
    if odometer_sampling_method is None:
        odometer_sampling_method = config.odometer_sampling_method
    if odometer_sampling_bounds is None:
        odometer_sampling_bounds = config.odometer_sampling_bounds

    resolved_odometer_stats_path = _resolve_odometer_stats_path(
        odometer_stats_path,
        instance_name=INSTANCE,
    )
    if resolved_odometer_stats_path is not None:
        initialized_bikes = apply_component_odometer_initialization(
            state,
            stats_path=resolved_odometer_stats_path,
            rng=state.rng_degradation,
            method=odometer_sampling_method,
            bounds=odometer_sampling_bounds,
        )
        print(
            f"Initialized component odometers for {initialized_bikes} bikes from "
            f"{resolved_odometer_stats_path} ({odometer_sampling_method}, {odometer_sampling_bounds})."
        )
        _log_debug_bike_odometer_initialization(
            state=state,
            seed=seed,
            bike_ids=config.odometer_debug_bike_ids,
            stats_path=resolved_odometer_stats_path,
            sampling_method=odometer_sampling_method,
            sampling_bounds=odometer_sampling_bounds,
        )
    elif str(odometer_stats_path).lower() == AUTO_ODOMETER_STATS:
        print("No steady-state odometer stats found; starting component odometers at zero.")
    
    FLEET_SIZE = state.get_all_bikes()
    print(f"Initialized state with {len(FLEET_SIZE)} bikes for instance '{INSTANCE}' and seed {seed}.")



    # Ensure the simulator state clears any existing vehicles first
    if hasattr(state, 'vehicles'):
        state.vehicles = {} 

    # Create the list of policies for the vehicles
    # If the simulator creates a NEW Vehicle object for each entry in this list:
    vehicle_policies = [policy for _ in range(num_vehicles)]
    state.set_sb_vehicles(vehicle_policies)
 
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
        verbose=verbose,
    )
    simulator.operation_logger = operation_logger
    simulator.run_logger = run_logger

    if run_logger is not None:
        run_logger.capture_fleet_start(state)

    if operation_logger.enabled:
        print("[OPS] Operational logging enabled (step-by-step vehicle/action trace)")


    print(
        f"Running simulation with duration {duration}, vehicles {num_vehicles}, "
        f"seed {seed}, Instance {INSTANCE}"
    )

    if ENABLE_COMPONENT_FAILURES:
        print("Component failure simulation: ENABLED")

    policy.maintenance_enabled = MAINTENANCE_ENABLED

    simulator.run()
  
    if queue is not None:
        queue.put(simulator)
    return simulator


def write_simulation_outputs(simulator, filename, seed, policy, duration, num_vehicles, append_to_results=False, run_logger=None, write_csv=True):
    """Write all output files for a single simulation run.

    Args:
        simulator: The completed simulator instance
        filename: Base filename for results (e.g., 'policy_results.csv')
        seed: Random seed used for this simulation
        policy: Policy instance with weights
        duration: Simulation duration in hours
        num_vehicles: Number of vehicles used
        append_to_results: Whether to append to main results file (False for first seed)
        write_csv: Whether to write simulation_results/csv/ files (set False when RunLogger is the primary output)
    """
    solve_time = simulator.state.time

    if write_csv:
        base_filename = filename.replace('.csv', '')
        weights = getattr(policy, "weights", None)
        alpha_value = weights[3] if weights and len(weights) > 3 else None

        write_results_to_file(
            filename,
            simulator,
            duration,
            solve_time,
            seed,
            append=append_to_results,
            run_logger=run_logger,
        )

        daily_metrics_filename = f"{base_filename}_daily_metrics_seed_{seed}.csv"
        write_daily_metrics_to_file(daily_metrics_filename, simulator, seed)
    

        bike_movements_filename = f"{base_filename}_bike_movements_seed_{seed}.csv"
        write_bike_movements_to_file(bike_movements_filename, simulator, seed, alpha_value)

        if ENABLE_COMPONENT_FAILURES:
            component_failures_filename = f"{base_filename}_component_failures_seed_{seed}.csv"
            #write_component_failures_to_file(component_failures_filename, simulator, seed, alpha_value)
            #write_component_failures_to_file(component_failures_filename, simulator, seed, alpha_value)

        vehicle_health_log_filename = f"{base_filename}_vehicle_health_seed_{seed}.csv"
        write_vehicle_and_health_logs(vehicle_health_log_filename, simulator, seed)

        # rl_decisions_filename = f"{base_filename}_rl_decisions_seed_{seed}.csv"
        # write_rl_decisions_to_file(rl_decisions_filename, simulator, seed)
        # rl_decisions_filename = f"{base_filename}_rl_decisions_seed_{seed}.csv"
        # write_rl_decisions_to_file(rl_decisions_filename, simulator, seed)

    if run_logger is not None:
        run_logger.log_episode(simulator, seed, duration, solve_time)

    print(f"Seed {seed}: Completed in {solve_time:.2f}s")

def test_seeds(
    list_of_seeds,
    policy,
    filename,
    num_vehicles=1,
    duration=24*5,
    use_multiprocessing=True,
    instance_name=None,
    config=None,
    run_logger=None,
    write_csv=True,
    odometer_stats_path=None,
    odometer_sampling_method="triangular",
    odometer_sampling_bounds="p05-p95",
):
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
        write_csv: Whether to write simulation_results/csv/ files
    """
    if config is None:
        config = SimulationConfig()
    
    results_file = filename
  
    if use_multiprocessing:
        # Run simulations in parallel
        queue = mp.Queue()
        processes = []
  
        for seed in list_of_seeds:
            p = mp.Process(
                target=run_simulation,
                args=(seed, policy, duration, num_vehicles, queue, instance_name, config),
                kwargs={
                    "odometer_stats_path": odometer_stats_path,
                    "odometer_sampling_method": odometer_sampling_method,
                    "odometer_sampling_bounds": odometer_sampling_bounds,
                },
            )
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
                append_to_results=(i > 0),
                write_csv=write_csv,
            )

    else:
        # Run simulations sequentially (easier for debugging)
        for i, seed in enumerate(list_of_seeds):
            print(f"\nRunning seed {seed}...")
            if run_logger is not None:
                run_logger.set_seed(seed)
            start_solve = time.time()
            simulator = run_simulation(
                seed,
                policy,
                duration,
                num_vehicles,
                instance_name=instance_name,
                config=config,
                run_logger=run_logger,
                odometer_stats_path=odometer_stats_path,
                odometer_sampling_method=odometer_sampling_method,
                odometer_sampling_bounds=odometer_sampling_bounds,
            )
            solve_time = time.time() - start_solve

            write_simulation_outputs(
                simulator=simulator,
                filename=results_file,
                seed=seed,
                policy=policy,
                duration=duration,
                num_vehicles=num_vehicles,
                append_to_results=(i > 0),
                run_logger=run_logger,
                write_csv=write_csv,
            )

    if write_csv:
        print(f"\nResults written to: policies/sjovik_sund/simulation_results/{results_file}")
 
 
def test_policies(
    list_of_seeds,
    policy_dict,
    num_vehicles=1,
    duration=24*5,
    use_multiprocessing=False,
    instance_name=None,
    config=None,
    run_logger=None,
    write_csv=True,
    odometer_stats_path=None,
    odometer_sampling_method="triangular",
    odometer_sampling_bounds="p05-p95",
):
    """Test multiple policies with multiple seeds.

    Args:
        list_of_seeds: List of random seeds to test
        policy_dict: Dictionary of policy_name -> policy instance
        num_vehicles: Number of vehicles
        duration: Simulation duration in hours
        use_multiprocessing: Whether to run in parallel
        instance_name: Instance name
        config: SimulationConfig instance
        write_csv: Whether to write simulation_results/csv/ files
    """
    if config is None:
        config = SimulationConfig()

    for policy_name, policy in policy_dict.items():
        print(f"\n{'='*80}")
        print(f"Testing Policy: {policy_name}")
        print(f"{'='*80}\n")

        results_file = f'{policy_name}_results.csv'
        test_seeds(
            list_of_seeds,
            policy,
            results_file,
            num_vehicles,
            duration,
            use_multiprocessing,
            instance_name,
            config,
            run_logger=run_logger,
            write_csv=write_csv,
            odometer_stats_path=odometer_stats_path,
            odometer_sampling_method=odometer_sampling_method,
            odometer_sampling_bounds=odometer_sampling_bounds,
        )


def generate_steady_state_odometer_stats(
    list_of_seeds,
    duration,
    num_vehicles=1,
    instance_name=None,
    config=None,
    output_dir=None,
):
    """Run GreedyMaintenancePolicy and export steady-state component odometer stats."""
    if config is None:
        config = SimulationConfig()
    if instance_name is None:
        instance_name = config.default_instance

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = (
            WORKSPACE_ROOT
            / "policies"
            / "sjovik_sund"
            / "simulation_results"
            / "steady_state_odometers"
            / f"{instance_name}_D{duration}h_V{num_vehicles}_{timestamp}"
        )
    output_dir = Path(output_dir)

    all_seed_rows = []
    all_sample_rows = []
    for seed in list_of_seeds:
        print(
            f"\n[steady-state odometers] seed={seed}, policy=GreedyMaintenancePolicy, "
            f"duration={duration}h"
        )
        simulator = run_simulation(
            seed=seed,
            policy=GreedyMaintenancePolicy(),
            duration=duration,
            num_vehicles=num_vehicles,
            instance_name=instance_name,
            config=config,
            run_logger=None,
            verbose=False,
        )
        all_seed_rows.extend(summarize_component_odometers(simulator.state, seed=seed))
        all_sample_rows.extend(collect_component_odometer_samples(simulator.state, seed=seed))

    aggregate_rows = aggregate_seed_summaries(all_seed_rows)

    per_seed_path = output_dir / "component_odometer_stats_by_seed.csv"
    aggregate_path = output_dir / "component_odometer_stats_aggregated.csv"
    sample_path = output_dir / "component_odometer_samples.csv"

    write_csv_rows(per_seed_path, all_seed_rows)
    write_csv_rows(aggregate_path, aggregate_rows)
    write_csv_rows(sample_path, all_sample_rows)

    print(f"\nWrote per-seed odometer stats to: {per_seed_path}")
    print(f"Wrote aggregated odometer stats to: {aggregate_path}")
    print(f"Wrote raw odometer samples to: {sample_path}")
    return aggregate_path
 
 
if __name__ == "__main__":
    # Create centralized configuration
    config = SimulationConfig()
    
    # ---- argument parsing for CLI ----
    parser = argparse.ArgumentParser(
        description="Run Sjovik & Sund simulation with different alpha values."
    )
    parser.add_argument(
        "--policy",
        type=str,
        nargs="+",
        choices=["vfa", "hybrid", "greedy-maintenance", "greedy-v2", "greedy", "do-nothing", "xpilot"],
        default=["vfa"],
        help=(
            "Policy or policies to run sequentially (default: vfa). "
            "Example: --policy vfa greedy-maintenance."
        ),
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
        help=f"Simulation duration in hours (default: {config.default_duration_hours} hours = 56 days / 8 weeks).",
    )
    parser.add_argument(
        "--maintenance_limit",
        type=float,
        default=config.default_maintenance_limit,
        help=f"Maintenance criticality limit to check for maintenance actions (default: {config.default_maintenance_limit}).",
    )
    parser.add_argument(
        "--vfa-model",
        type=str,
        default="models/final_ablation_500ep_timefix/Imbalance_Squared_Temporal_alpha_0.1_20260429_202323/vfa_Imbalance_Squared_Temporal_seed1000.pkl",
        help="Path to trained VFA model (.pkl file).",
    )
    parser.add_argument(
        "--active-features",
        type=str,
        nargs="+",
        default=[
            "rebalancing_imbalance",
            "squared_starvation_penalty",
            "squared_congestion_penalty",
            "gross_starvation_risk",
            "gross_congestion_risk",
        ],
        help=(
            "Feature names the pkl was trained with. Required for old pkls that "
            "predate feature_names storage. New pkls load features automatically."
        ),
    )
    parser.add_argument(
        "--num-scenarios", "--scenarios",
        dest="num_scenarios",
        type=int,
        default=8,
        help="Number of stochastic scenarios per candidate in HybridRolloutPolicy (default: 8).",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=60.0,
        help="Hybrid rollout horizon in minutes (default: 60).",
    )
    parser.add_argument(
        "--n-time-steps",
        type=int,
        default=12,
        help="Number of analytical demand sub-steps in the rollout horizon (default: 12).",
    )
    parser.add_argument(
        "--n-routing",
        type=int,
        default=10,
        help="Routing candidates per operational profile in candidate generation (default: 10).",
    )
    parser.add_argument(
        "--rollout-degradation",
        action="store_true",
        default=False,
        help="Sample component failures inside the analytical rollout horizon.",
    )
    parser.add_argument(
        "--hybrid-debug",
        action="store_true",
        default=False,
        help="Print Hybrid rollout candidate tables at each decision.",
    )
    parser.add_argument(
        "--log-files",
        nargs="+",
        choices=["results", "hourly", "daily", "decisions", "debug", "all", "none"],
        default=None,
        help=(
            "Centralized run_logs outputs to write. Examples: "
            "'--log-files results daily', '--log-files all', '--log-files none'. "
            "Default: results."
        ),
    )
    parser.add_argument(
        "--run-logger",
        action="store_true",
        default=False,
        help="Deprecated alias for '--log-files results hourly daily debug'.",
    )
    parser.add_argument(
        "--log-decisions",
        action="store_true",
        default=False,
        help="Deprecated alias that adds 'decisions' to --log-files.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        default=True,
        help="Skip legacy simulation_results/csv writers and rely on centralized run_logs output.",
    )
    parser.add_argument(
        "--generate-odometer-stats",
        action="store_true",
        default=False,
        help=(
            "Run GreedyMaintenancePolicy for the selected seeds and write steady-state "
            "component odometer CSVs, then exit."
        ),
    )
    parser.add_argument(
        "--odometer-stats-out",
        type=str,
        default=None,
        help="Output directory for --generate-odometer-stats CSVs.",
    )
    parser.add_argument(
        "--odometer-stats-in",
        type=str,
        default=None,
        help=(
            "CSV with aggregated component odometer stats used to warm-start bikes "
            "before a normal simulation run."
        ),
    )
    parser.add_argument(
        "--odometer-sampling-method",
        type=str,
        choices=["triangular", "uniform", "truncated-normal"],
        default="triangular",
        help="Sampling method used with --odometer-stats-in (default: triangular).",
    )
    parser.add_argument(
        "--odometer-sampling-bounds",
        type=str,
        choices=["p05-p95", "min-max"],
        default="p05-p95",
        help=(
            "Bounds used when sampling warm-start odometers. p05-p95 is more robust; "
            "min-max uses the full observed range."
        ),
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

    if args.generate_odometer_stats:
        generate_steady_state_odometer_stats(
            list_of_seeds=list_of_seeds,
            duration=duration,
            num_vehicles=num_vehicles,
            instance_name=args.instance,
            config=config,
            output_dir=args.odometer_stats_out,
        )
        sys.exit(0)

    timestamp = datetime.now().strftime("%m%d%H%M")

    def _resolve_log_files() -> list[str]:
        if args.log_files is not None:
            selected = list(args.log_files)
        elif args.run_logger:
            selected = ["results", "hourly", "daily", "debug"]
        else:
            selected = ["results"]

        if args.log_decisions:
            if "none" in selected:
                selected = ["decisions"]
            elif "all" not in selected and "decisions" not in selected:
                selected.append("decisions")
        return selected

    log_files = _resolve_log_files()
    run_logger = None if "none" in log_files else SimulationRunLogger(log_files=log_files)

    def _set_run_logger_context(policy_type: str, exp_name: str = "", alpha: float = 0.0, results_only: bool = False) -> None:
        if run_logger is None:
            return
        run_logger.set_run_label(
            exp_name,
            alpha,
            policy_type=policy_type,
            duration_hours=duration,
            num_vehicles=num_vehicles,
            instance_name=args.instance,
            results_only=results_only,
        )

    def _load_vfa_policy() -> tuple[LinearVFAPolicy, str]:
        load_kwargs = {}
        if args.active_features:
            load_kwargs["active_features"] = args.active_features
        vfa_policy = LinearVFAPolicy.load(Path(args.vfa_model), **load_kwargs)
        vfa_policy.learning_mode = False
        import re
        stem = Path(args.vfa_model).stem          # e.g. "vfa_Squared_Temporal_seed1000"
        exp_name = re.sub(r"^vfa_|_seed\d+$", "", stem)  # e.g. "Squared_Temporal"
        return vfa_policy, exp_name

    def _build_policy(policy_type: str):
        if policy_type == "do-nothing":
            _set_run_logger_context(policy_type)
            policy_name = f"DoNothing_{args.instance}_V{num_vehicles}_D{duration}h_{timestamp}_seed{start_seed}"
            return policy_name, DoNothing()

        if policy_type == "greedy":
            _set_run_logger_context(policy_type)
            policy_name = f"Greedy_{args.instance}_V{num_vehicles}_D{duration}h_{timestamp}_seed{start_seed}"
            return policy_name, GreedyPolicy()

        if policy_type == "greedy-v2":
            _set_run_logger_context(policy_type)
            policy_name = f"GreedyV2_{args.instance}_V{num_vehicles}_D{duration}h_{timestamp}_seed{start_seed}"
            return policy_name, GreedyPolicyV2()

        if policy_type == "greedy-maintenance":
            _set_run_logger_context(policy_type)
            policy_name = f"GreedyMaintenance_{args.instance}_V{num_vehicles}_D{duration}h_{timestamp}_seed{start_seed}"
            policy = GreedyMaintenancePolicy()
            if run_logger is not None:
                policy.logger = run_logger
            return policy_name, policy

        if policy_type == "xpilot":
            _set_run_logger_context(policy_type)
            policy_name = f"XPILOT_{args.instance}_V{num_vehicles}_D{duration}h_{timestamp}_seed{start_seed}"
            policy = policies.sjovik_sund.XPILOT_policy.XPILOTPolicy(
                time_horizon=40, max_depth=2, num_successors=5, number_of_scenarios=100
            )
            return policy_name, policy

        if policy_type == "vfa":
            vfa_policy, exp_name = _load_vfa_policy()
            _set_run_logger_context(policy_type, exp_name=exp_name)
            if run_logger is not None:
                vfa_policy.logger = run_logger
            policy_name = (
                f"VFA_{exp_name}_{args.instance}_V{num_vehicles}_D{duration}h_"
                f"{timestamp}_seed{start_seed}"
            )
            return policy_name, vfa_policy

        if policy_type == "hybrid":
            vfa_policy, exp_name = _load_vfa_policy()
            _set_run_logger_context(policy_type, exp_name=exp_name)
            if run_logger is not None:
                vfa_policy.logger = run_logger
            hybrid_policy = HybridRolloutPolicy(
                trained_vfa=vfa_policy,
                lookahead_minutes=args.lookahead,
                num_scenarios=args.num_scenarios,
                n_routing_candidates=args.n_routing,
                n_time_steps=args.n_time_steps,
                use_degradation=args.rollout_degradation,
                logger=run_logger,
                debug_print=args.hybrid_debug,
            )
            policy_name = (
                f"Hybrid_{exp_name}_H{int(args.lookahead)}_S{args.num_scenarios}_"
                f"{args.instance}_V{num_vehicles}_D{duration}h_{timestamp}_seed{start_seed}"
            )
            return policy_name, hybrid_policy

        raise ValueError(f"Unknown policy type: {policy_type}")

    # Start timing
    start_time = time.time()
 
    # Test policies sequentially so each policy gets its own run_logger folder.
    for policy_type in args.policy:
        policy_name, policy = _build_policy(policy_type)
        test_policies(
            list_of_seeds=list_of_seeds,
            policy_dict={policy_name: policy},
            num_vehicles=num_vehicles,
            duration=duration,
            use_multiprocessing=False,
            instance_name=args.instance,
            config=config,
            run_logger=run_logger,
            write_csv=not args.no_csv,
            odometer_stats_path=args.odometer_stats_in,
            odometer_sampling_method=args.odometer_sampling_method,
            odometer_sampling_bounds=args.odometer_sampling_bounds,
        )

    if run_logger is not None:
        run_logger.close()
        print(f"Centralized logging output written to: {run_logger.run_dir}")
 
    # End timing
    total_duration = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total running time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"{'='*80}\n")
