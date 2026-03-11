"""
VFA State Representation for DSJBRMP

This module defines the state space for the MDP formulation:
- Station inventories (functional, onsite-repairable, depot-severe)
- Vehicle status (location, capacity, current load)
- Post-decision state transformations
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import numpy as np


@dataclass
class StationInventory:
    """
    Station bike inventory by maintenance state.
    
    Attributes:
        functional: Number of functional bikes (rentable)
        onsite: Number of bikes needing on-site repair
        depot: Number of bikes needing depot removal
        capacity: Total docking capacity of the station
        station_id: Station identifier
    """
    functional: int
    onsite: int
    depot: int
    capacity: int
    station_id: str
    
    def __post_init__(self):
        """Validate capacity constraint."""
        total = self.functional + self.onsite + self.depot
        if total > self.capacity:
            raise ValueError(
                f"Station {self.station_id}: total bikes ({total}) exceeds capacity ({self.capacity})"
            )
    
    def total_bikes(self) -> int:
        """Total bikes at station."""
        return self.functional + self.onsite + self.depot
    
    def available_docks(self) -> int:
        """Number of free docking spaces."""
        return self.capacity - self.total_bikes()
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """Return as (functional, onsite, depot) tuple."""
        return (self.functional, self.onsite, self.depot)
    
    def copy(self):
        """Create a deep copy."""
        return StationInventory(
            functional=self.functional,
            onsite=self.onsite,
            depot=self.depot,
            capacity=self.capacity,
            station_id=self.station_id
        )


@dataclass
class VehicleStatus:
    """
    Service vehicle state.
    
    Attributes:
        vehicle_id: Vehicle identifier
        current_station: Current/destination station ID
        arrival_time: Arrival time at current_station (continuous time)
        functional_cargo: Number of functional bikes in vehicle
        depot_cargo: Number of severe bikes being transported to depot
        capacity: Vehicle capacity
    """
    vehicle_id: int
    current_station: str
    arrival_time: float
    functional_cargo: int
    depot_cargo: int
    capacity: int
    
    def __post_init__(self):
        """Validate capacity constraint."""
        total_cargo = self.functional_cargo + self.depot_cargo
        if total_cargo > self.capacity:
            raise ValueError(
                f"Vehicle {self.vehicle_id}: total cargo ({total_cargo}) exceeds capacity ({self.capacity})"
            )
    
    def total_cargo(self) -> int:
        """Total bikes being transported."""
        return self.functional_cargo + self.depot_cargo
    
    def available_capacity(self) -> int:
        """Remaining vehicle capacity."""
        return self.capacity - self.total_cargo()
    
    def is_at_station(self, current_time: float) -> bool:
        """Check if vehicle has arrived at its destination."""
        return current_time >= self.arrival_time
    
    def copy(self):
        """Create a deep copy."""
        return VehicleStatus(
            vehicle_id=self.vehicle_id,
            current_station=self.current_station,
            arrival_time=self.arrival_time,
            functional_cargo=self.functional_cargo,
            depot_cargo=self.depot_cargo,
            capacity=self.capacity
        )


@dataclass
class MDPState:
    """
    Complete MDP state S_k at decision epoch k.
    
    This represents the pre-decision state when a vehicle arrives at a station.
    """
    time: float
    active_vehicle_id: int
    stations: Dict[str, StationInventory]
    vehicles: Dict[int, VehicleStatus]
    
    def get_station(self, station_id: str) -> StationInventory:
        """Get station inventory by ID."""
        return self.stations[station_id]
    
    def get_vehicle(self, vehicle_id: int) -> VehicleStatus:
        """Get vehicle status by ID."""
        return self.vehicles[vehicle_id]
    
    def get_active_vehicle(self) -> VehicleStatus:
        """Get the currently active (decision-making) vehicle."""
        return self.vehicles[self.active_vehicle_id]
    
    def copy(self):
        """Create a deep copy of the state."""
        return MDPState(
            time=self.time,
            active_vehicle_id=self.active_vehicle_id,
            stations={sid: inv.copy() for sid, inv in self.stations.items()},
            vehicles={vid: veh.copy() for vid, veh in self.vehicles.items()}
        )


@dataclass
class Action:
    """
    Action x = (ι, m_rep, m_rem, ρ)
    
    Attributes:
        rebalancing: ι^x - positive for delivery, negative for pickup
        onsite_repairs: m_rep^x - number of bikes repaired on-site
        depot_removals: m_rem^x - number of severe bikes removed to depot
        next_station: ρ^x - next station to route to
        station_id: Station where action is performed
    """
    rebalancing: int  # Can be negative (pickup) or positive (delivery)
    onsite_repairs: int
    depot_removals: int
    next_station: str
    station_id: str
    
    def __post_init__(self):
        """Validate action constraints."""
        if self.onsite_repairs < 0:
            raise ValueError("On-site repairs cannot be negative")
        if self.depot_removals < 0:
            raise ValueError("Depot removals cannot be negative")


class PostDecisionState:
    """
    Post-decision state S_k^x after applying action x.
    
    This is the deterministic state immediately after the action,
    before any stochastic events (demand/returns) occur.
    """
    
    @staticmethod
    def apply_action(state: MDPState, action: Action) -> MDPState:
        """
        Apply action to state to produce post-decision state.
        
        Args:
            state: Pre-decision state S_k
            action: Action x to apply
        
        Returns:
            Post-decision state S_k^x
        """
        # Create a copy to avoid modifying original state
        post_state = state.copy()
        
        # Get the active vehicle and target station
        vehicle = post_state.get_active_vehicle()
        station = post_state.get_station(action.station_id)
        
        # Verify vehicle is at the station
        if vehicle.current_station != action.station_id:
            raise ValueError(
                f"Vehicle {vehicle.vehicle_id} is at {vehicle.current_station}, "
                f"not at action station {action.station_id}"
            )
        
        # === Apply Action Components ===
        
        # 1. On-site repairs: onsite -> functional
        if action.onsite_repairs > station.onsite:
            raise ValueError(
                f"Cannot repair {action.onsite_repairs} bikes, "
                f"only {station.onsite} onsite bikes available"
            )
        station.onsite -= action.onsite_repairs
        station.functional += action.onsite_repairs
        
        # 2. Depot removals: depot -> vehicle depot cargo
        if action.depot_removals > station.depot:
            raise ValueError(
                f"Cannot remove {action.depot_removals} bikes, "
                f"only {station.depot} depot bikes available"
            )
        if action.depot_removals > vehicle.available_capacity():
            raise ValueError(
                f"Cannot load {action.depot_removals} depot bikes, "
                f"vehicle only has {vehicle.available_capacity()} capacity"
            )
        station.depot -= action.depot_removals
        vehicle.depot_cargo += action.depot_removals
        
        # 3. Rebalancing
        if action.rebalancing > 0:
            # Delivery: vehicle functional cargo -> station functional
            if action.rebalancing > vehicle.functional_cargo:
                raise ValueError(
                    f"Cannot deliver {action.rebalancing} bikes, "
                    f"vehicle only has {vehicle.functional_cargo}"
                )
            if action.rebalancing > station.available_docks():
                raise ValueError(
                    f"Cannot deliver {action.rebalancing} bikes, "
                    f"station only has {station.available_docks()} free docks"
                )
            vehicle.functional_cargo -= action.rebalancing
            station.functional += action.rebalancing
        
        elif action.rebalancing < 0:
            # Pickup: station functional -> vehicle functional cargo
            pickup_amount = -action.rebalancing
            if pickup_amount > station.functional:
                raise ValueError(
                    f"Cannot pickup {pickup_amount} bikes, "
                    f"station only has {station.functional} functional bikes"
                )
            if pickup_amount > vehicle.available_capacity():
                raise ValueError(
                    f"Cannot pickup {pickup_amount} bikes, "
                    f"vehicle only has {vehicle.available_capacity()} capacity"
                )
            station.functional -= pickup_amount
            vehicle.functional_cargo += pickup_amount
        
        # 4. Update vehicle routing
        vehicle.current_station = action.next_station
        # Note: arrival_time will be updated externally based on travel time
        
        return post_state
    
    @staticmethod
    def validate_constraints(state: MDPState) -> bool:
        """
        Verify all state constraints are satisfied.
        
        Returns:
            True if all constraints are valid, raises exception otherwise
        """
        for station_id, station in state.stations.items():
            if station.total_bikes() > station.capacity:
                raise ValueError(
                    f"Station {station_id} exceeds capacity: "
                    f"{station.total_bikes()} > {station.capacity}"
                )
        
        for vehicle_id, vehicle in state.vehicles.items():
            if vehicle.total_cargo() > vehicle.capacity:
                raise ValueError(
                    f"Vehicle {vehicle_id} exceeds capacity: "
                    f"{vehicle.total_cargo()} > {vehicle.capacity}"
                )
        
        return True


class StateObservationWrapper:
    """
    Wrapper to extract MDP state from the high-fidelity simulator.
    
    The simulator tracks individual Bike objects with component odometers
    and Weibull failure distributions. This wrapper aggregates them into
    the clean integer tuples (f^func, f^onsite, f^depot) for the VFA.
    """
    
    @staticmethod
    def extract_station_inventory(station) -> StationInventory:
        """
        Extract station inventory from simulator Station object.
        
        Args:
            station: Simulator Station object
        
        Returns:
            StationInventory with aggregated bike counts
        """
        functional = 0
        onsite = 0
        depot = 0
        
        # Iterate through all bikes at the station
        bikes = station.get_bikes()  # Get all bikes at this station
        
        for bike in bikes:
            # Aggregate based on damage_status
            if hasattr(bike, 'damage_status'):
                if bike.damage_status == "depot":
                    depot += 1
                elif bike.damage_status == "onsite":
                    onsite += 1
                elif bike.damage_status is None:
                    # No damage - functional
                    functional += 1
                else:
                    # Unknown damage status - treat as functional for safety
                    functional += 1
            else:
                # Bike has no damage tracking - treat as functional
                functional += 1
        
        return StationInventory(
            functional=functional,
            onsite=onsite,
            depot=depot,
            capacity=station.capacity,
            station_id=station.id
        )
    
    @staticmethod
    def extract_vehicle_status(vehicle, simul_time: float) -> VehicleStatus:
        """
        Extract vehicle status from simulator Vehicle object.
        
        Args:
            vehicle: Simulator Vehicle object
            simul_time: Current simulation time
        
        Returns:
            VehicleStatus with current state
        """
        functional_cargo = 0
        depot_cargo = 0
        
        # Count bikes in vehicle cargo by damage status
        if hasattr(vehicle, 'bikes'):
            for bike in vehicle.bikes:
                if hasattr(bike, 'damage_status'):
                    if bike.damage_status == "depot":
                        depot_cargo += 1
                    else:
                        # Onsite or None - treat as functional cargo
                        functional_cargo += 1
                else:
                    functional_cargo += 1
        
        return VehicleStatus(
            vehicle_id=vehicle.id,
            current_station=vehicle.current_station.id if hasattr(vehicle, 'current_station') else None,
            arrival_time=simul_time,  # Simplified - vehicle is at station
            functional_cargo=functional_cargo,
            depot_cargo=depot_cargo,
            capacity=vehicle.capacity
        )
    
    @staticmethod
    def extract_mdp_state(simul, active_vehicle_id: int) -> MDPState:
        """
        Extract complete MDP state from simulator.
        
        Args:
            simul: Simulator object
            active_vehicle_id: ID of vehicle making decision
        
        Returns:
            Complete MDPState
        """
        # Extract all station inventories
        stations = {}
        for station in simul.state.get_stations():
            stations[station.id] = StateObservationWrapper.extract_station_inventory(station)
        
        # Extract all vehicle statuses
        vehicles = {}
        for vehicle_id, vehicle in simul.state.vehicles.items():
            vehicles[vehicle_id] = StateObservationWrapper.extract_vehicle_status(
                vehicle, simul.state.time
            )
        
        return MDPState(
            time=simul.state.time,
            active_vehicle_id=active_vehicle_id,
            stations=stations,
            vehicles=vehicles
        )


class StochasticTransition:
    """
    Represents stochastic information ω_k between decision epochs.
    
    This captures:
    - Successful rentals (ψ_k^n)
    - Returns by damage state (ω_k^{n,func}, ω_k^{n,onsite}, ω_k^{n,depot})
    """
    
    @dataclass
    class StationEvent:
        """Events at a single station."""
        station_id: str
        rentals: int  # Successful rentals (departed)
        returns_functional: int
        returns_onsite: int
        returns_depot: int
        failed_rentals: int  # Demand when station empty
        failed_returns: int  # Returns when station full
    
    def __init__(self):
        self.events: Dict[str, StochasticTransition.StationEvent] = {}
    
    def add_station_event(self, event: 'StochasticTransition.StationEvent'):
        """Record events at a station."""
        self.events[event.station_id] = event
    
    def get_total_failed_rentals(self) -> int:
        """Total failed rentals across all stations."""
        return sum(e.failed_rentals for e in self.events.values())
    
    def get_total_failed_returns(self) -> int:
        """Total failed returns across all stations."""
        return sum(e.failed_returns for e in self.events.values())
