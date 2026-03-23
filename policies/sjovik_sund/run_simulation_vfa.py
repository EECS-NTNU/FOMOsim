# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Get workspace root (2 levels up from this file)
WORKSPACE_ROOT = Path(__file__).parents[2]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, '')

# Force Python to not use cached bytecode
sys.dont_write_bytecode = True
import importlib
if hasattr(importlib, 'invalidate_caches'):
    importlib.invalidate_caches()

import init_state
import target_state
import sim
import demand
from helpers import timeInMinutes
from settings import *

# Import only the necessary logging utilities
from policies.sjovik_sund.simulation_logging import (
    RESULTS_DIR,
    LoggingSimulator,
    write_results_to_file,
    write_hourly_metrics_to_file,
    write_station_hourly_metrics_to_file,
    write_component_failures_to_file,
    write_vehicle_and_health_logs, 
    write_rl_decisions_to_file,
)
from policies.sjovik_sund.operational_logging import OperationalLogger

from dataclasses import dataclass, field
from typing import Dict, List

#MAINTENANCE_ENABLED = False


@dataclass
class SimulationConfig:
    """Centralized configuration for VFA simulation runs."""
    
    # === Time Settings ===
    start_hour: int = 7  # 7 AM start time
    
    # === Instance Settings ===
    default_instance: str = "TD_W34_old"
    
    # === Instance-Specific Start Stations ===
    instance_start_stations: Dict[str, List[int]] = field(default_factory=lambda: {
        "TD": [0, 5, 10, 15, 20, 25, 30, 35, 40],
        "OS": [4, 5, 10, 15, 20, 25, 30, 35, 40],
        "EH": [0, 1, 2, 3, 4, 5, 6, 7] * 3 + [0, 1, 2]
    })
    
    # === Operational Debug Logging ===
    operation_logging_enabled: bool = True
    operation_logging_include_bike_ids: bool = True
    
    # === Target State ===
    target_state_type: str = "half_capacity"
    
    def get_start_stations(self, instance_name: str) -> List[int]:
        """Get start stations list based on instance name prefix."""
        for prefix, stations in self.instance_start_stations.items():
            if prefix in instance_name:
                return stations
        return [0]  # Fallback default
    
    def get_target_state_instance(self):
        """Return the appropriate target state instance."""
        if self.target_state_type == "half_capacity":
            return target_state.HalfCapacityTargetState()
        elif self.target_state_type == "equal_prob":
            return target_state.EqualProbTargetState()
        elif self.target_state_type == "us":
            return target_state.USTargetState()
        else:
            return target_state.HalfCapacityTargetState()


def run_simulation(seed, policy, duration=24, num_vehicles=1, instance_name=None, config=None):
    """Run a single simulation for the VFA agent."""
    if config is None:
        config = SimulationConfig()
    
    START_TIME = timeInMinutes(hours=config.start_hour)
    DURATION = timeInMinutes(hours=duration)
 
    if instance_name is None:
        instance_name = config.default_instance
    
    INSTANCE = instance_name

    # Load initial state
    instance_path = WORKSPACE_ROOT / "instances" / INSTANCE
    state = init_state.read_initial_state(str(instance_path))
    state.set_seed(seed)

    #if MAINTENANCE_ENABLED:
        #state.initialize_bike_maintenance()

    # Clear any existing vehicles
    if hasattr(state, 'vehicles'):
        state.vehicles = {} 

    # Assign the VFA policy to the vehicle(s)
    vehicle_policies = [policy for _ in range(num_vehicles)]
    state.set_sb_vehicles(vehicle_policies)
 
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

    # Step-by-step operational logger
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

    #policy.maintenance_enabled = MAINTENANCE_ENABLED
  
    simulator.run()
    return simulator


def write_simulation_outputs(simulator, filename, seed, policy, duration, num_vehicles, append_to_results=False):
    """Write VFA output files for a single simulation run into organized folders."""
    solve_time = simulator.state.time
    base_filename = filename.replace('.csv', '')
    
    # Build the folder structure
    target_dir = (RESULTS_DIR / base_filename).parent
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Main aggregated results
    write_results_to_file(filename, simulator, duration, solve_time, seed, append=append_to_results)
    
    # 2. Hourly network metrics
    hourly_filename = f"{base_filename}_hourly_seed_{seed}.csv"
    write_hourly_metrics_to_file(hourly_filename, simulator, seed)
    
    # 3. Station hourly metrics
    station_hourly_filename = f"{base_filename}_station_hourly_seed_{seed}.csv"
    write_station_hourly_metrics_to_file(station_hourly_filename, simulator, seed)
    
    # 4. Component Failures (if enabled)
    if ENABLE_COMPONENT_FAILURES:
        component_failures_filename = f"{base_filename}_component_failures_seed_{seed}.csv"
        write_component_failures_to_file(component_failures_filename, simulator, seed)

    # 5. Vehicle Cargo & EOD Health
    vehicle_health_log_filename = f"{base_filename}_vehicle_health_seed_{seed}.csv"
    write_vehicle_and_health_logs(vehicle_health_log_filename, simulator, seed)

    # 6. RL Agent Brain Decisions
    rl_decisions_filename = f"{base_filename}_rl_decisions_seed_{seed}.csv"
    write_rl_decisions_to_file(rl_decisions_filename, simulator, seed)
    
    print(f"Seed {seed}: Completed in {solve_time:.2f}s. Results saved to {target_dir}")