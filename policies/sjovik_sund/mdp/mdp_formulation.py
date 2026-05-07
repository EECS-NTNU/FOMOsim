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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .mdp_config import MDPConfig

from settings import MINUTES_PER_ACTION, MAINTENANCE_REPAIR, MINUTES_CONSTANT_PER_ACTION


# ─────────────────────────────────────────────────────────────────────────────
# Station inventory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StationInventory:
    """
    Inventory at a normal (non-depot) station.

    f_k^n = (functional, onsite, depot)
    """
    station_id:  str
    functional:  int   # rentable bikes
    onsite:      int   # bikes needing on-site repair (repairable without depot trip)
    depot:       int   # bikes requiring removal to depot
    capacity:    int   # docking capacity
    target:      int   = 0    # target inventory for this station at current time-of-day
    expected_departure_rate: float = 0.0  # bikes/hour leaving this station (from demand model)
    expected_arrival_rate:   float = 0.0  # bikes/hour arriving at this station (from demand model)

    def total_bikes(self) -> int:
        return self.functional + self.onsite + self.depot

    def free_docks(self) -> int:
        return self.capacity - self.total_bikes()

    def to_tuple(self) -> Tuple[int, int, int]:
        """(functional, onsite, depot)"""
        return self.functional, self.onsite, self.depot


@dataclass
class DepotInventory:
    """
    Inventory at the depot station (n_0).
    
    The depot operates differently: broken bikes are unloaded for repair,
    finished bikes are picked up from a queue. No capacity constraints.
    
    Attributes
    ──────────
    station_id   : depot identifier (e.g., "n_0")
    fixed_queue  : bikes finished repair, ready to pick up
    in_repair    : bikes currently in 24h repair cycle
    capacity     : effectively unlimited (set to large value for compatibility)
    """
    station_id: str
    fixed_queue: int = 0  # finished bikes ready to load onto vehicle
    in_repair: int = 0    # bikes currently in 24h repair (not available for pickup)
    capacity: int = int(1e9)  # unlimited

    def total_bikes(self) -> int:
        """Bikes at depot (fixed_queue + in_repair)."""
        return self.fixed_queue + self.in_repair

    def free_docks(self) -> int:
        """Unlimited at depot."""
        return int(1e9)


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
    stations          : station_id → StationInventory (normal stations)
    depot             : DepotInventory or None (if depot exists)
    vehicles          : vehicle_id → VehicleStatus  (all vehicles in fleet)
    config            : MDPConfig – scenario mode (damage tracking, maintenance control)
    shift_end_time    : end of shift (minutes); used to compute time_remaining for end-of-day planning
    """
    time:               float
    active_vehicle_id:  int
    stations:           Dict[str, StationInventory]
    depot:              Optional[DepotInventory]
    vehicles:           Dict[int,  VehicleStatus]
    config:             MDPConfig = field(default_factory=MDPConfig.full_maintenance)
    shift_end_time:     Optional[float] = None  # end of shift for the active vehicle (minutes)
    travel_times:       Optional[Dict[str, float]] = None  # {station_id: minutes from active vehicle}

    def get_station(self, station_id: str) -> StationInventory:
        """Get a normal station (not depot)."""
        return self.stations[station_id]

    def get_vehicle(self, vehicle_id: int) -> VehicleStatus:
        return self.vehicles[vehicle_id]

    def get_active_vehicle(self) -> VehicleStatus:
        return self.vehicles[self.active_vehicle_id]
    
    def is_at_depot(self, station_id: str) -> bool:
        """Check if station_id is the depot."""
        return self.depot is not None and self.depot.station_id == station_id
    
    def time_remaining_in_shift(self) -> float:
        """
        Compute remaining time in shift (minutes).
        
        Returns:
            time_remaining : float
                max(0, shift_end_time - current_time)
                If shift_end_time is None, returns a large value (shift is effectively infinite).
        """
        if self.shift_end_time is None:
            return float(1e9)  # no hard end-of-shift
        return max(0.0, self.shift_end_time - self.time)
    
    def shift_time_fraction_remaining(self) -> float:
        """
        Normalized time remaining as a fraction of the standard 24-hour shift.

        Returns:
            frac : float in [0, 1]
                0 = shift over, 1.0 = shift entirely ahead
                If shift_end_time is None, returns 1.0.

        The denominator is a fixed 1440-minute reference (24 hours), matching
        LinearVFAPolicy._get_shift_length().  Using remaining time as the
        denominator would always return 1.0 — which was the previous bug.
        """
        if self.shift_end_time is None:
            return 1.0
        remaining = self.time_remaining_in_shift()   # max(0, shift_end_time - time)
        return min(1.0, max(0.0, remaining / 1440.0))


# ─────────────────────────────────────────────────────────────────────────────
# Action  x = (ι, m_rep, m_rem, ρ)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MdpAction:
    """
    Decision made by the active vehicle at a station.

    Semantics differ between normal stations and depot:

    At normal stations:
        rebalancing    : >0 deliver functional bikes, <0 pick up functional bikes
        onsite_repairs : bikes to repair in-place (onsite → functional)
        depot_removals : damaged bikes to load for depot transport
        load_from_queue: must be 0 (not at depot)

    At depot:
        rebalancing    : must be 0 (no delivery/pickup at depot)
        onsite_repairs : must be 0
        depot_removals : must be 0 (broken bikes auto-unload on arrival)
        depot_dropoffs : depot-damaged cargo unloaded for repair
        load_from_queue: bikes to pick from fixed_queue (repair-finished bikes)
    """
    current_station:  str
    rebalancing:      int  # ι^x  (signed; normal station only)
    onsite_repairs:   int  # m_rep^x  (normal station only)
    depot_removals:   int  # m_rem^x  (normal station only)
    load_from_queue:  int  # bikes from fixed_queue (depot only)
    next_station:     str  # ρ^x
    depot_dropoffs:   int = 0  # depot-damaged cargo unloaded at depot


# ─────────────────────────────────────────────────────────────────────────────
# Executed action record (for logging and policy visibility)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutedAction:
    """
    Record of what actually happened as a result of an action.
    
    Used for logging, cost breakdown, and explicit reward calculation in policy.
    """
    bikes_repaired_onsite: int = 0       # at normal station
    bikes_removed_to_depot: int = 0      # loaded for depot transport
    bikes_picked_up: int = 0             # functional bikes picked up
    bikes_unloaded_for_repair: int = 0   # at depot: broken bikes → in-repair
    bikes_loaded_from_queue: int = 0     # at depot: from fixed_queue
    labor_minutes: float = 0.0           # total time spent


# ─────────────────────────────────────────────────────────────────────────────
# Post-decision state  S_k^x
# ─────────────────────────────────────────────────────────────────────────────

class PostDecisionState:
    """
    Compute the deterministic post-decision state S_k^x.

    PURPOSE: purely for evaluating V̄(S_k^x) = θᵀφ(S_k^x) and rollout
    lookahead.  Never modifies the live simulator.

    Provides two separate apply methods:
    - apply_at_normal_station() : handles rebalancing and maintenance actions
    - apply_at_depot() : handles automatic unload + selective load from queue

    Both return (new_state, action_duration_minutes) so the caller can compute:
        new_eta = current_time + action_duration + travel_time_to_next_station
    """

    # Settings constants (import from settings module)
    #MINUTES_PER_ACTION = MINUTES_PER_ACTION      # time to load/unload per bike
    #MAINTENANCE_REPAIR = MAINTENANCE_REPAIR      # time to fully repair one bike on-site

    @staticmethod
    def apply(state: MDPState, action: MdpAction) -> Tuple[MDPState, float, ExecutedAction]:
        """
        Dispatch to normal station or depot handler.
        
        Returns (new_state, action_duration_minutes, executed_action)
        """
        v = state.get_active_vehicle()
        
        if state.is_at_depot(action.current_station):
            return PostDecisionState.apply_at_depot(state, action)
        else:
            return PostDecisionState.apply_at_normal_station(state, action)

    @staticmethod
    def apply_at_normal_station(state: MDPState, action: MdpAction) -> Tuple[MDPState, float, ExecutedAction]:
        """
        Apply action at a normal (non-depot) station.
        
        Returns (new_state, action_duration_minutes, executed_action)
        """
        v = state.get_active_vehicle()
        s = state.get_station(action.current_station)
        cfg = state.config

        # ── validate ──────────────────────────────────────────────────────
        if action.load_from_queue != 0:
            raise ValueError(
                f"At normal station {action.current_station}: "
                f"load_from_queue must be 0 (only used at depot)"
            )
        if action.depot_dropoffs != 0:
            raise ValueError(
                f"At normal station {action.current_station}: "
                f"depot_dropoffs must be 0 (only used at depot)"
            )
        if action.onsite_repairs < 0:
            raise ValueError("onsite_repairs must be ≥ 0")
        if action.depot_removals < 0:
            raise ValueError("depot_removals must be ≥ 0")

        if not cfg.allow_onsite_repairs and action.onsite_repairs != 0:
            raise ValueError("onsite repairs disabled by MDPConfig")
        if not cfg.allow_depot_removals and action.depot_removals != 0:
            raise ValueError("depot removals disabled by MDPConfig")

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

        # Check combined depot_removals + pickup do not exceed free_capacity
        if action.rebalancing < 0:
            pickup = -action.rebalancing
            if action.depot_removals + pickup > v.free_capacity():
                raise ValueError(
                    f"Cannot load {action.depot_removals} depot bikes "
                    f"+ pick up {pickup} functional bikes; "
                    f"total {action.depot_removals + pickup} exceeds "
                    f"vehicle capacity {v.free_capacity()}"
                )
        else:
            # Delivery (rebalancing > 0) unloads bikes first, freeing that many slots
            effective_free = v.free_capacity() + max(0, action.rebalancing)
            if action.depot_removals > effective_free:
                raise ValueError(
                    f"Cannot load {action.depot_removals} bikes; "
                    f"vehicle only has {effective_free} free capacity "
                    f"(current={v.free_capacity()} + delivery={max(0, action.rebalancing)})"
                )

        # Rebalancing delivery bounds
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
        # Rebalancing pickup bounds
        elif action.rebalancing < 0:
            pickup = -action.rebalancing
            if pickup > s.functional:
                raise ValueError(
                    f"Cannot pick up {pickup} bikes; "
                    f"station has {s.functional} functional"
                )

        # ── build new station inventory ────────────────────────────────────
        onsite_rep = action.onsite_repairs if cfg.allow_onsite_repairs else 0
        depot_rem = action.depot_removals if cfg.allow_depot_removals else 0

        new_functional = s.functional + onsite_rep
        new_onsite = s.onsite - onsite_rep
        new_depot_s = s.depot - depot_rem

        if action.rebalancing > 0:
            new_functional += action.rebalancing
        elif action.rebalancing < 0:
            new_functional += action.rebalancing

        if not cfg.track_damage:
            new_onsite = 0
            new_depot_s = 0

        new_station = StationInventory(
            station_id=s.station_id,
            functional=new_functional,
            onsite=new_onsite,
            depot=new_depot_s,
            capacity=s.capacity,
            target=s.target,   # propagate target — unchanged by action
            expected_departure_rate=s.expected_departure_rate,
            expected_arrival_rate=s.expected_arrival_rate,
        )

        # ── build new vehicle status ───────────────────────────────────────
        new_func_cargo = v.functional_cargo - max(0, action.rebalancing) \
                         + max(0, -action.rebalancing)
        new_depot_cargo = v.depot_cargo + depot_rem
        if not cfg.track_damage:
            new_depot_cargo = 0

        # ── compute action duration ────────────────────────────────────────
        # Time = unload depot bikes + move functional bikes (either direction) + on-site repairs
        time_unload_depot = depot_rem * MINUTES_PER_ACTION
        time_rebalancing = abs(action.rebalancing) * MINUTES_PER_ACTION
        time_onsite_repairs = onsite_rep * MAINTENANCE_REPAIR
        action_duration = time_unload_depot + time_rebalancing + time_onsite_repairs

        # ── compute travel time to next station ───────────────────────────
        travel_time_to_next = state.travel_times.get(action.next_station, 0.0) if state.travel_times else 0.0
        new_eta = state.time + action_duration + travel_time_to_next

        new_vehicle = VehicleStatus(
            vehicle_id=v.vehicle_id,
            destination_station=action.next_station,
            eta=new_eta,
            functional_cargo=new_func_cargo,
            depot_cargo=new_depot_cargo,
            capacity=v.capacity,
        )

        # ── build executed action record ───────────────────────────────────────────────────
        executed = ExecutedAction(
            bikes_repaired_onsite=onsite_rep,
            bikes_removed_to_depot=depot_rem,
            bikes_picked_up=max(0, -action.rebalancing),
            labor_minutes=action_duration,
        )
        # ── assemble post-decision state ─────────────────────────────────
        new_stations = {**state.stations, action.current_station: new_station}
        new_vehicles = {**state.vehicles, v.vehicle_id: new_vehicle}

        new_state = MDPState(
            time=state.time,
            active_vehicle_id=state.active_vehicle_id,
            stations=new_stations,
            depot=state.depot,
            vehicles=new_vehicles,
            config=cfg,
            shift_end_time=state.shift_end_time,
            travel_times=state.travel_times,   # static distances — unchanged by action
        )

        return new_state, action_duration, executed

    @staticmethod
    def apply_at_depot(state: MDPState, action: MdpAction) -> Tuple[MDPState, float, ExecutedAction]:
        """
        Apply action at the depot station.
        
        At depot:
        - All depot-damaged cargo must be unloaded for repair
        - Functional bikes (if any) remain on vehicle
        - Vehicle can load bikes from fixed_queue (repair-finished bikes)
        
        Returns (new_state, action_duration_minutes, executed_action)
        """
        v = state.get_active_vehicle()
        depot = state.depot
        cfg = state.config
        if depot is None:
            raise ValueError("Cannot apply depot action: state has no depot configured")
 
        # ── validate ──────────────────────────────────────────────────────
        if action.rebalancing != 0:
            raise ValueError(
                f"At depot {depot.station_id}: rebalancing must be 0 "
                f"(use load_from_queue instead)"
            )
        if action.onsite_repairs != 0:
            raise ValueError(
                f"At depot {depot.station_id}: onsite_repairs must be 0"
            )
        if action.depot_removals != 0:
            raise ValueError(
                f"At depot {depot.station_id}: depot_removals must be 0 "
                f"(broken bikes auto-unload on arrival)"
            )
        if action.load_from_queue < 0:
            raise ValueError("load_from_queue must be ≥ 0")
        if action.load_from_queue > depot.fixed_queue:
            raise ValueError(
                f"Cannot load {action.load_from_queue} bikes from fixed_queue; "
                f"only {depot.fixed_queue} available"
            )
        depot_dropoffs = action.depot_dropoffs
        if depot_dropoffs != v.depot_cargo:
            raise ValueError(
                f"At depot {depot.station_id}: depot_dropoffs must unload all "
                f"depot cargo ({v.depot_cargo}); got {depot_dropoffs}"
            )

        free_capacity_after_unload = max(0, v.capacity - v.functional_cargo - (v.depot_cargo - depot_dropoffs))
        if action.load_from_queue > free_capacity_after_unload:
            raise ValueError(
                f"Cannot load {action.load_from_queue} bikes; "
                f"after unloading depot cargo, vehicle has "
                f"{free_capacity_after_unload} free slots"
            )
 
        # ── build new depot inventory ──────────────────────────────────────
        # Broken bikes are moved to in-repair
        bikes_entering_repair = depot_dropoffs
        new_in_repair = depot.in_repair + bikes_entering_repair
        new_fixed_queue = depot.fixed_queue - action.load_from_queue
 
        new_depot = DepotInventory(
            station_id=depot.station_id,
            fixed_queue=new_fixed_queue,
            in_repair=new_in_repair,
            capacity=depot.capacity,
        )
 
        # ── build new vehicle status ───────────────────────────────────────
        # After unload: vehicle only has functional_cargo + newly loaded bikes from queue
        new_func_cargo = v.functional_cargo + action.load_from_queue
        new_depot_cargo = v.depot_cargo - depot_dropoffs
 
        new_vehicle = VehicleStatus(
            vehicle_id=v.vehicle_id,
            destination_station=action.next_station,
            eta=state.time,  # will be updated by caller
            functional_cargo=new_func_cargo,
            depot_cargo=new_depot_cargo,
            capacity=v.capacity,
        )
 
        # ── compute action duration ────────────────────────────────────────
        # Time = unload all broken bikes + load bikes from queue
        time_unload_broken = depot_dropoffs * MINUTES_PER_ACTION
        time_load_from_queue = action.load_from_queue * MINUTES_PER_ACTION
        action_duration = MINUTES_CONSTANT_PER_ACTION + time_unload_broken + time_load_from_queue
        # NOTE: Repair duration (24h cycle) is not added here—it's exogenous,
        # managed by the simulator between decision epochs.
        # ── build executed action record ───────────────────────────────────────────────────
        executed = ExecutedAction(
            bikes_unloaded_for_repair=depot_dropoffs,
            bikes_loaded_from_queue=action.load_from_queue,
            labor_minutes=action_duration,
        )
        # ── assemble post-decision state ─────────────────────────────────
        new_vehicles = {**state.vehicles, v.vehicle_id: new_vehicle}
 
        new_state = MDPState(
            time=state.time,
            active_vehicle_id=state.active_vehicle_id,
            stations=state.stations,  # normal stations unchanged
            depot=new_depot,
            vehicles=new_vehicles,
            config=cfg,
            shift_end_time=state.shift_end_time,
            travel_times=state.travel_times,   # static distances — unchanged by action
        )
 
        return new_state, action_duration, executed
 
 
 


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
    
    def total_bike_breakages(self) -> int:
        return sum(e.returns_onsite + e.returns_depot for e in self.events.values())


# ─────────────────────────────────────────────────────────────────────────────
# State extraction from the live simulator
# ─────────────────────────────────────────────────────────────────────────────

def _count_bikes(station, config: Optional[MDPConfig] = None) -> Tuple[int, int, int]:
    """
    Count (functional, onsite, depot) bikes at a simulator Station.

    If config.track_damage is False, returns (total_bikes, 0, 0).
    Otherwise iterates the bike list once; all other MDP computations are
    numpy-vectorised over the resulting arrays.
    """
    cfg = config or MDPConfig.full_maintenance()

    if not cfg.track_damage:
        return len(station.bikes), 0, 0

    functional = onsite = depot = 0
    for bike in station.bikes.values():
        ds = getattr(bike, "damage_status", None)
        if ds == "depot":
            depot += 1
        elif ds == "onsite":
            onsite += 1
        else:
            functional += 1
    return functional, onsite, depot


def extract_station_inventory(
    station,
    config: Optional[MDPConfig] = None,
    target: int = 0,
    expected_departure_rate: float = 0.0,
    expected_arrival_rate:   float = 0.0,
) -> StationInventory:
    """
    Build a StationInventory from a live sim.Station object.

    Args:
        station                : sim.Station object (normal operating station, not depot)
        config                 : MDPConfig (defaults to full maintenance)
        target                 : target inventory for this station at current time-of-day
        expected_departure_rate: expected departures/hour from demand model (bikes/hour)
        expected_arrival_rate  : expected arrivals/hour from demand model (bikes/hour)
    """
    cfg = config or MDPConfig.full_maintenance()
    func, onsite, depot = _count_bikes(station, cfg)
    return StationInventory(
        station_id=station.id,
        functional=func,
        onsite=onsite,
        depot=depot,
        capacity=station.capacity,
        target=target,
        expected_departure_rate=expected_departure_rate,
        expected_arrival_rate=expected_arrival_rate,
    )


def extract_depot_inventory(
    depot_station,
    config: Optional[MDPConfig] = None,
) -> DepotInventory:
    """
    Build a DepotInventory from the depot sim.Station object.

    Args:
        depot_station : sim.Station object representing the depot (e.g., "n_0")
        config        : MDPConfig (defaults to full maintenance)
    """
    cfg = config or MDPConfig.full_maintenance()
    # Extract fixed_queue and in_repair from depot station attributes
    fixed_queue = len(getattr(depot_station, "fixed_queue", []))
    in_repair   = sum(len(bikes) for _, bikes in getattr(depot_station, "in_repair", []))
    
    return DepotInventory(
        station_id=depot_station.id,
        fixed_queue=fixed_queue,
        in_repair=in_repair,
        capacity=depot_station.capacity,
    )


def extract_vehicle_status(
    vehicle,
    current_time: float,
    config: Optional[MDPConfig] = None
) -> VehicleStatus:
    """
    Build a VehicleStatus from a live sim.Vehicle object.

    The vehicle is assumed to be at (or en route to) its current_station;
    eta is set to current_time when the vehicle is already at the station.

    Args:
        vehicle      : sim.Vehicle
        current_time : sim.State.time (minutes)
        config       : MDPConfig (defaults to full maintenance)
    """
    cfg = config or MDPConfig.full_maintenance()

    if hasattr(vehicle, "get_bike_inventory"):
        bikes_on_vehicle = list(vehicle.get_bike_inventory())
    else:
        raw_inventory = getattr(vehicle, "bike_inventory", {})
        if isinstance(raw_inventory, dict):
            bikes_on_vehicle = list(raw_inventory.values())
        else:
            bikes_on_vehicle = list(raw_inventory)

    functional_cargo = depot_cargo = 0
    for bike in bikes_on_vehicle:
        ds = getattr(bike, "damage_status", None)
        if cfg.track_damage and ds == "depot":
            depot_cargo += 1
        else:
            functional_cargo += 1

    # destination_station: use the vehicle's current location id
    location = getattr(vehicle, "location", None)
    dest = getattr(location, "id", None)
    if dest is None:
        current_station = getattr(vehicle, "current_station", None)
        dest = getattr(current_station, "id", current_station)
    if dest is None:
        raise ValueError(f"Cannot extract vehicle {vehicle.id}: missing current location")
    dest_id = str(dest)

    return VehicleStatus(
        vehicle_id=vehicle.id,
        destination_station=dest_id,
        eta=current_time,
        functional_cargo=functional_cargo,
        depot_cargo=depot_cargo if cfg.track_damage else 0,
        capacity=getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", len(bikes_on_vehicle))),
    )


def extract_mdp_state(
    sim_state,
    active_vehicle_id: int,
    config: Optional[MDPConfig] = None,
    depot_id: Optional[str] = None,
    shift_end_time: Optional[float] = None,
) -> MDPState:
    """
    Build a full MDPState snapshot from the live sim.State.

    This is a read-only projection – the simulator is not modified.

    Args:
        sim_state          : sim.State  (the .state attribute of Simulator)
        active_vehicle_id  : ID of the vehicle currently making a decision
        config             : MDPConfig (defaults to full maintenance)
        depot_id           : ID of the depot station (e.g. "n_0"), if any
        shift_end_time     : end-of-shift time (minutes); if provided, enables
                             end-of-day anticipatory behavior in the VFA
    """
    cfg = config or MDPConfig.full_maintenance()

    # Extract normal stations (sim.State keeps depots in a separate dict,
    # so get_stations() never contains the depot — look it up via get_depots()).
    stations = {}
    depot = None
    sim_day  = sim_state.day()
    sim_hour = sim_state.hour()
    for s in sim_state.get_stations():
        try:
            target = round(s.get_target_state(sim_day, sim_hour))
        except Exception:
            target = 0
        stations[s.id] = extract_station_inventory(
            s, cfg, target=target,
            expected_departure_rate=s.get_leave_intensity(sim_day, sim_hour),
            expected_arrival_rate=s.get_arrive_intensity(sim_day, sim_hour),
        )

    for d in sim_state.get_depots():
        if depot_id is None or d.id == depot_id:
            depot = extract_depot_inventory(d, cfg)
            if depot_id is None:
                depot_id = d.id  # use first depot found
            break

    vehicles = {
        v.id: extract_vehicle_status(v, sim_state.time, cfg)
        for v in sim_state.get_vehicles()
    }

    # Travel times from the active vehicle's current location to every station.
    # The active vehicle has just arrived, so destination_station == current location.
    travel_times = None
    active_loc = vehicles[active_vehicle_id].destination_station if active_vehicle_id in vehicles else None
    if active_loc is not None:
        travel_times = {
            sid: sim_state.get_vehicle_travel_time(active_loc, sid)
            for sid in stations
        }
        if depot_id is not None:
            try:
                travel_times[depot_id] = sim_state.get_vehicle_travel_time(active_loc, depot_id)
            except (KeyError, Exception):
                pass  # depot not in locations dict for this sim instance

    return MDPState(
        time=sim_state.time,
        active_vehicle_id=active_vehicle_id,
        stations=stations,
        depot=depot,
        vehicles=vehicles,
        config=cfg,
        shift_end_time=shift_end_time,
        travel_times=travel_times,
    )
