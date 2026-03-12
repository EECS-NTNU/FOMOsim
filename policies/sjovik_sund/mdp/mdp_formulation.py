"""
mdp_formulation.py  –  MDP Formulation for DSJBRMP

Defines the state space, action space, and post-decision state transition
for the Dynamic Stochastic Joint Bike Rebalancing and Maintenance Problem.

This module is algorithm-agnostic: both the VFA (LinearVFAPolicy) and the
future Rollout Algorithm import from here.

─────────────────────────────────────────────────────────────────────────────
State  S_k  at decision epoch k  (vehicle arrives at station, time t_k)
─────────────────────────────────────────────────────────────────────────────
  Station inventory  f_k^n = (f_k^{n,func}, f_k^{n,onsite}, f_k^{n,depot})
    functional : rentable bikes
    onsite     : bikes needing on-site repair (repairable without depot trip)
    depot      : bikes requiring removal to the depot

  Vehicle status  V_k^v  (one entry per service vehicle v)
    destination_station : n_k^v – station the vehicle is en route to, or has
                                   just arrived at (string ID)
    eta                 : α_k^v – scheduled arrival time in simulation minutes;
                                   equals t_k for the vehicle making the decision
    functional_cargo    : functional bikes currently on the vehicle
    depot_cargo         : depot-level damaged bikes being transported
    capacity            : total vehicle capacity (constant)

─────────────────────────────────────────────────────────────────────────────
Action  x = (ι^x, m_rep^x, m_rem^x, ρ^x)  at station n, vehicle v
─────────────────────────────────────────────────────────────────────────────
  rebalancing    (ι^x)   : >0 deliver functional bikes, <0 pick up
  onsite_repairs (m_rep^x): bikes repaired in-place (onsite → functional)
  depot_removals (m_rem^x): damaged bikes loaded onto vehicle (depot → cargo)
  next_station   (ρ^x)   : ID of the station the vehicle routes to next

─────────────────────────────────────────────────────────────────────────────
Post-decision state  S_k^x
─────────────────────────────────────────────────────────────────────────────
  Deterministic state immediately after action x, before stochastic events.
  Used exclusively for VFA evaluation and rollout lookahead – does NOT
  modify the live simulator.

─────────────────────────────────────────────────────────────────────────────
Stochastic transition  ω_k  (between epochs k and k+1)
─────────────────────────────────────────────────────────────────────────────
  Captures observed demand and returns at each station.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Station inventory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StationInventory:
    """
    Aggregated bike inventory at one station.

    f_k^n = (functional, onsite, depot)
    """
    station_id:  str
    functional:  int   # rentable bikes
    onsite:      int   # bikes needing on-site repair
    depot:       int   # bikes requiring depot removal
    capacity:    int   # total docking capacity

    def total_bikes(self) -> int:
        return self.functional + self.onsite + self.depot

    def free_docks(self) -> int:
        return self.capacity - self.total_bikes()

    def to_tuple(self) -> Tuple[int, int, int]:
        """(functional, onsite, depot)"""
        return self.functional, self.onsite, self.depot


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle status
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleStatus:
    """
    Status of one service vehicle v at decision epoch k.

    Supports fleets with multiple vehicles – MDPState holds one entry per
    vehicle, keyed by vehicle_id.

    Attributes
    ──────────
    vehicle_id          : unique vehicle identifier
    destination_station : n_k^v – station the vehicle is en route to,
                          or has just arrived at
    eta                 : α_k^v – arrival time (simulation minutes);
                          equal to t_k for the vehicle currently making a
                          decision (i.e. it has just arrived)
    functional_cargo    : functional bikes on board
    depot_cargo         : depot-level damaged bikes on board
    capacity            : vehicle capacity (constant)
    """
    vehicle_id:          int
    destination_station: str
    eta:                 float   # α_k^v
    functional_cargo:    int
    depot_cargo:         int
    capacity:            int

    def total_cargo(self) -> int:
        return self.functional_cargo + self.depot_cargo

    def free_capacity(self) -> int:
        return self.capacity - self.total_cargo()

    def has_arrived(self, current_time: float) -> bool:
        """True when the vehicle has reached its destination."""
        return current_time >= self.eta


# ─────────────────────────────────────────────────────────────────────────────
# MDP state  S_k
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MDPState:
    """
    Complete MDP pre-decision state S_k at epoch k.

    Attributes
    ──────────
    time              : t_k – current simulation time (minutes)
    active_vehicle_id : ID of the vehicle that just arrived and is deciding
    stations          : station_id → StationInventory
    vehicles          : vehicle_id → VehicleStatus  (all vehicles in fleet)
    """
    time:               float
    active_vehicle_id:  int
    stations:           Dict[str, StationInventory]
    vehicles:           Dict[int,  VehicleStatus]

    def get_station(self, station_id: str) -> StationInventory:
        return self.stations[station_id]

    def get_vehicle(self, vehicle_id: int) -> VehicleStatus:
        return self.vehicles[vehicle_id]

    def get_active_vehicle(self) -> VehicleStatus:
        return self.vehicles[self.active_vehicle_id]


# ─────────────────────────────────────────────────────────────────────────────
# Action  x = (ι, m_rep, m_rem, ρ)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MdpAction:
    """
    Decision made by the active vehicle at station station_id.

    Attributes
    ──────────
    station_id     : station where the action is performed
    rebalancing    : ι^x  – >0 deliver functional bikes, <0 pick up
    onsite_repairs : m_rep^x – bikes repaired in-place (onsite → functional)
    depot_removals : m_rem^x – damaged bikes loaded onto vehicle (depot → cargo)
    next_station   : ρ^x  – ID of the next station to route to
    """
    station_id:     str
    rebalancing:    int   # ι^x  (signed)
    onsite_repairs: int   # m_rep^x  (≥ 0)
    depot_removals: int   # m_rem^x  (≥ 0)
    next_station:   str   # ρ^x


# ─────────────────────────────────────────────────────────────────────────────
# Post-decision state  S_k^x
# ─────────────────────────────────────────────────────────────────────────────

class PostDecisionState:
    """
    Compute the deterministic post-decision state S_k^x.

    PURPOSE: purely for evaluating V̄(S_k^x) = θᵀφ(S_k^x) and rollout
    lookahead.  Never modifies the live simulator.

    All methods construct NEW dataclass instances from the affected fields;
    no deepcopy is performed.
    """

    @staticmethod
    def apply(state: MDPState, action: MdpAction) -> MDPState:
        """
        Return S_k^x: the state after applying action x to pre-decision
        state S_k.

        Only the station being visited and the active vehicle change;
        all other objects are reused directly (no copies made).

        Args:
            state  : pre-decision MDPState S_k
            action : MdpAction x

        Returns:
            post-decision MDPState S_k^x
        """
        v    = state.get_active_vehicle()
        s    = state.get_station(action.station_id)

        # ── validate ──────────────────────────────────────────────────────
        if action.onsite_repairs < 0:
            raise ValueError("onsite_repairs must be ≥ 0")
        if action.depot_removals < 0:
            raise ValueError("depot_removals must be ≥ 0")
        if action.onsite_repairs > s.onsite:
            raise ValueError(
                f"Cannot repair {action.onsite_repairs} bikes; "
                f"only {s.onsite} onsite at {s.station_id}"
            )
        if action.depot_removals > s.depot:
            raise ValueError(
                f"Cannot remove {action.depot_removals} depot bikes; "
                f"only {s.depot} at {s.station_id}"
            )
        if action.depot_removals > v.free_capacity():
            raise ValueError(
                f"Cannot load {action.depot_removals} bikes; "
                f"vehicle only has {v.free_capacity()} free capacity"
            )

        # Rebalancing bounds
        if action.rebalancing > 0:
            if action.rebalancing > v.functional_cargo:
                raise ValueError(
                    f"Cannot deliver {action.rebalancing} bikes; "
                    f"vehicle has {v.functional_cargo}"
                )
            if action.rebalancing > s.free_docks():
                raise ValueError(
                    f"Cannot deliver {action.rebalancing} bikes; "
                    f"station has {s.free_docks()} free docks"
                )
        elif action.rebalancing < 0:
            pickup = -action.rebalancing
            if pickup > s.functional:
                raise ValueError(
                    f"Cannot pick up {pickup} bikes; "
                    f"station has {s.functional} functional"
                )
            if pickup > v.free_capacity():
                raise ValueError(
                    f"Cannot pick up {pickup} bikes; "
                    f"vehicle has {v.free_capacity()} free capacity"
                )

        # ── build new station inventory ────────────────────────────────────
        new_functional  = s.functional  + action.onsite_repairs
        new_onsite      = s.onsite      - action.onsite_repairs
        new_depot_s     = s.depot       - action.depot_removals

        if action.rebalancing > 0:
            new_functional += action.rebalancing
        elif action.rebalancing < 0:
            new_functional += action.rebalancing   # subtracts (rebalancing < 0)

        new_station = StationInventory(
            station_id=s.station_id,
            functional=new_functional,
            onsite=new_onsite,
            depot=new_depot_s,
            capacity=s.capacity,
        )

        # ── build new vehicle status ───────────────────────────────────────
        new_func_cargo  = v.functional_cargo - max(0,  action.rebalancing) \
                          + max(0, -action.rebalancing)
        new_depot_cargo = v.depot_cargo + action.depot_removals

        new_vehicle = VehicleStatus(
            vehicle_id=v.vehicle_id,
            destination_station=action.next_station,
            eta=state.time,   # will be updated by caller with actual travel time
            functional_cargo=new_func_cargo,
            depot_cargo=new_depot_cargo,
            capacity=v.capacity,
        )

        # ── assemble post-decision state (reuse unchanged objects) ─────────
        new_stations = {**state.stations, action.station_id: new_station}
        new_vehicles = {**state.vehicles, v.vehicle_id: new_vehicle}

        return MDPState(
            time=state.time,
            active_vehicle_id=state.active_vehicle_id,
            stations=new_stations,
            vehicles=new_vehicles,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic transition  ω_k
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StationEvent:
    """Observed demand/return events at one station during interval [t_k, t_{k+1})."""
    station_id:        str
    rentals:           int   # ψ_k^n   – successful rentals
    returns_functional: int   # ω_k^{n,func}
    returns_onsite:    int   # ω_k^{n,onsite}
    returns_depot:     int   # ω_k^{n,depot}
    failed_rentals:    int   # demand when station had 0 functional bikes
    failed_returns:    int   # returns when station was at capacity


class StochasticTransition:
    """
    Stochastic information ω_k observed between epochs k and k+1.

    Aggregates StationEvent objects for the full network.
    """

    def __init__(self) -> None:
        self.events: Dict[str, StationEvent] = {}

    def add(self, event: StationEvent) -> None:
        self.events[event.station_id] = event

    def total_failed_rentals(self) -> int:
        return sum(e.failed_rentals for e in self.events.values())

    def total_failed_returns(self) -> int:
        return sum(e.failed_returns for e in self.events.values())


# ─────────────────────────────────────────────────────────────────────────────
# State extraction from the live simulator
# ─────────────────────────────────────────────────────────────────────────────

def _count_bikes(station) -> Tuple[int, int, int]:
    """
    Count (functional, onsite, depot) bikes at a simulator Station.

    Iterates the bike list once; all other MDP computations are numpy-
    vectorised over the resulting arrays.
    """
    functional = onsite = depot = 0
    for bike in station.bikes:
        ds = getattr(bike, "damage_status", None)
        if ds == "depot":
            depot += 1
        elif ds == "onsite":
            onsite += 1
        else:
            functional += 1
    return functional, onsite, depot


def extract_station_inventory(station) -> StationInventory:
    """
    Build a StationInventory from a live sim.Station object.

    Args:
        station : sim.Station
    """
    func, onsite, depot = _count_bikes(station)
    return StationInventory(
        station_id=station.id,
        functional=func,
        onsite=onsite,
        depot=depot,
        capacity=station.capacity,
    )


def extract_vehicle_status(vehicle, current_time: float) -> VehicleStatus:
    """
    Build a VehicleStatus from a live sim.Vehicle object.

    The vehicle is assumed to be at (or en route to) its current_station;
    eta is set to current_time when the vehicle is already at the station.

    Args:
        vehicle      : sim.Vehicle
        current_time : sim.State.time (minutes)
    """
    functional_cargo = depot_cargo = 0
    for bike in getattr(vehicle, "bikes", []):
        ds = getattr(bike, "damage_status", None)
        if ds == "depot":
            depot_cargo += 1
        else:
            functional_cargo += 1

    # destination_station: use the vehicle's current location id
    dest = getattr(vehicle.location, "id", None) or getattr(vehicle, "current_station", None)
    if hasattr(dest, "id"):   # unwrap if it's a Station object
        dest = dest.id

    return VehicleStatus(
        vehicle_id=vehicle.id,
        destination_station=dest,
        eta=current_time,
        functional_cargo=functional_cargo,
        depot_cargo=depot_cargo,
        capacity=vehicle.capacity,
    )


def extract_mdp_state(sim_state, active_vehicle_id: int) -> MDPState:
    """
    Build a full MDPState snapshot from the live sim.State.

    This is a read-only projection – the simulator is not modified.

    Args:
        sim_state          : sim.State  (the .state attribute of Simulator)
        active_vehicle_id  : ID of the vehicle currently making a decision
    """
    stations = {
        s.id: extract_station_inventory(s)
        for s in sim_state.get_stations()
    }
    vehicles = {
        v.id: extract_vehicle_status(v, sim_state.time)
        for v in sim_state.get_vehicles()
    }
    return MDPState(
        time=sim_state.time,
        active_vehicle_id=active_vehicle_id,
        stations=stations,
        vehicles=vehicles,
    )
