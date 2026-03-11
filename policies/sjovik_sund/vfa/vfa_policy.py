"""
VFA Policy - Integration with Simulator

This policy uses the VFA agent to make rebalancing and maintenance decisions.
It bridges the gap between the MDP formulation and the simulator's action interface.

Key Implementation Details:
1. Action Space Splitting:
   - Micro-step: Optimize (ι, m_rep, m_rem) at current station
   - Macro-step: Evaluate routing choices (ρ) using VFA
   
2. Greedy Action Selection:
   - For computational efficiency, don't enumerate all joint actions
   - Instead: fix micro-decision greedily, then evaluate macro-decision
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from policies.policy import Policy
from sim.Action import Action
from typing import List, Tuple, Optional
import numpy as np

from .vfa_state import (
    MDPState, StationInventory, VehicleStatus, Action as MDPAction,
    PostDecisionState, StateObservationWrapper
)
from .vfa_features import VFAFeatures, DemandForecaster
from .vfa_agent import VFAAgent, LearningParameters


class VFAPolicy(Policy):
    """
    Policy using Value Function Approximation for joint rebalancing and maintenance.
    
    This policy learns optimal decisions through interaction with the simulator.
    """
    
    def __init__(self, 
                 vfa_agent: Optional[VFAAgent] = None,
                 learning_mode: bool = True,
                 maintenance_enabled: bool = True,
                 repair_time_per_bike: float = 5.0,  # minutes
                 seed: int = 42):
        """
        Initialize VFA policy.
        
        Args:
            vfa_agent: Pre-trained VFA agent (if None, creates new one)
            learning_mode: If True, update VFA during simulation
            maintenance_enabled: Whether to perform maintenance actions
            repair_time_per_bike: Time in minutes for on-site repair
            seed: Random seed
        """
        super().__init__(maintenance_enabled=maintenance_enabled)
        
        self.learning_mode = learning_mode
        self.repair_time_per_bike = repair_time_per_bike
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Initialize VFA agent if not provided
        if vfa_agent is None:
            forecaster = DemandForecaster()
            features = VFAFeatures(forecaster)
            learning_params = LearningParameters()
            self.vfa_agent = VFAAgent(features, learning_params, seed)
        else:
            self.vfa_agent = vfa_agent
        
        # Track previous state for learning updates
        self.prev_mdp_state = None
        self.prev_action = None
        self.prev_post_state = None
        
        # Statistics
        self.decision_count = 0
        self.update_count = 0
        
        print(f"\n{'='*80}")
        print(f"VFAPolicy initialized")
        print(f"  Learning mode: {self.learning_mode}")
        print(f"  Maintenance enabled: {self.maintenance_enabled}")
        print(f"  Feature dimensions: {self.vfa_agent.features.get_num_features()}")
        print(f"{'='*80}\n")
    
    def get_best_action(self, simul, vehicle) -> Action:
        """
        Get best action using VFA agent.
        
        This is called by the simulator when a vehicle needs to make a decision.
        
        Args:
            simul: Simulator object
            vehicle: Vehicle making the decision
        
        Returns:
            Action object for the simulator
        """
        self.decision_count += 1
        
        # === STEP 1: Extract MDP State ===
        mdp_state = StateObservationWrapper.extract_mdp_state(simul, vehicle.id)
        
        # === STEP 2: Update VFA from Previous Transition (if learning) ===
        if self.learning_mode and self.prev_post_state is not None:
            self._update_from_transition(simul, mdp_state)
        
        # === STEP 3: Generate Feasible Actions ===
        current_station_id = vehicle.current_station.id
        feasible_actions = self._generate_feasible_actions(
            mdp_state, current_station_id, vehicle
        )
        
        if len(feasible_actions) == 0:
            # No feasible actions - return do-nothing action
            print(f"Warning: No feasible actions for vehicle {vehicle.id} at {current_station_id}")
            return self._create_simulator_action(
                vehicle_id=vehicle.id,
                rebalancing=0,
                onsite_repairs=0,
                depot_removals=0,
                next_station=current_station_id,
                current_station=vehicle.current_station,
                simul=simul
            )
        
        # === STEP 4: Select Action ===
        if self.learning_mode:
            # Epsilon-greedy for exploration
            selected_action = self.vfa_agent.select_action_epsilon_greedy(
                mdp_state, feasible_actions
            )
        else:
            # Pure greedy for exploitation
            selected_action = self.vfa_agent.select_greedy_action(
                mdp_state, feasible_actions
            )
        
        # === STEP 5: Store for Next Update ===
        if self.learning_mode:
            self.prev_mdp_state = mdp_state
            self.prev_action = selected_action
            try:
                self.prev_post_state = PostDecisionState.apply_action(mdp_state, selected_action)
            except ValueError as e:
                print(f"Error applying action: {e}")
                self.prev_post_state = None
        
        # === STEP 6: Convert to Simulator Action ===
        simulator_action = self._mdp_action_to_simulator_action(
            selected_action, vehicle, simul
        )
        
        # Print decision info periodically
        if self.decision_count % 50 == 0:
            self.vfa_agent.print_learning_status()
        
        return simulator_action
    
    def _update_from_transition(self, simul, current_state: MDPState):
        """
        Update VFA parameters using observed transition.
        
        Implements TD update: θ ← θ - α∇[V̄(S^x) - (c + V̄(S'))]
        
        Args:
            simul: Simulator (for cost computation)
            current_state: Current state S_{k+1}
        """
        if self.prev_post_state is None:
            return
        
        # Compute observed cost from metrics
        observed_cost = self._compute_observed_cost(simul)
        
        # Perform TD update
        self.vfa_agent.update_theta(
            post_state=self.prev_post_state,
            observed_cost=observed_cost,
            next_state=current_state
        )
        
        self.update_count += 1
        
        # Decay exploration rate
        if self.update_count % 100 == 0:
            self.vfa_agent.decay_epsilon()
    
    def _compute_observed_cost(self, simul) -> float:
        """
        Compute observed cost from simulator metrics.
        
        Returns:
            Cost incurred since last decision
        """
        # Access failed events from metrics
        failed_rentals = 0
        failed_returns = 0
        
        # Try to get metrics from simulator
        if hasattr(simul.state, 'metrics'):
            metrics = simul.state.metrics
            
            # Get congestion metrics as proxy for failures
            if hasattr(metrics, 'get_aggregate_metric'):
                failed_rentals = metrics.get_aggregate_metric('short congestions', 0)
                failed_returns = metrics.get_aggregate_metric('roaming for locks', 0)
        
        cost = (self.vfa_agent.params.failed_rental_cost * failed_rentals +
                self.vfa_agent.params.failed_return_cost * failed_returns)
        
        return cost
    
    def _generate_feasible_actions(self, state: MDPState, 
                                   station_id: str, 
                                   vehicle) -> List[MDPAction]:
        """
        Generate feasible actions using action space splitting.
        
        Strategy:
        1. Micro-optimization: Enumerate (ι, m_rep, m_rem) combinations
        2. Macro-optimization: For each micro-action, consider routing options
        
        Args:
            state: Current MDP state
            station_id: Current station ID
            vehicle: Simulator vehicle object
        
        Returns:
            List of feasible MDPAction objects
        """
        feasible_actions = []
        
        station = state.get_station(station_id)
        vehicle_state = state.get_vehicle(vehicle.id)
        
        # === MICRO-STEP: Enumerate Local Actions ===
        # Rebalancing range: [-functional bikes, +functional cargo]
        max_pickup = min(station.functional, vehicle_state.available_capacity())
        max_delivery = min(vehicle_state.functional_cargo, station.available_docks())
        
        # Repair range: [0, onsite bikes]
        max_repairs = station.onsite
        
        # Removal range: [0, depot bikes] limited by vehicle capacity
        max_removals = min(station.depot, vehicle_state.available_capacity())
        
        # Generate micro-action combinations (sample to keep tractable)
        if max_pickup + max_delivery + max_repairs + max_removals == 0:
            # No local actions possible
            micro_actions = [(0, 0, 0)]
        else:
            micro_actions = self._sample_micro_actions(
                max_pickup, max_delivery, max_repairs, max_removals
            )
        
        # === MACRO-STEP: Routing Choices ===
        # Get candidate next stations (nearby stations)
        candidate_stations = self._get_candidate_routing_stations(
            simul=vehicle.current_station,  # Pass something with state access
            current_station_id=station_id
        )
        
        # === COMBINE: Generate Full Actions ===
        for rebal, repairs, removals in micro_actions:
            for next_station in candidate_stations:
                action = MDPAction(
                    rebalancing=rebal,
                    onsite_repairs=repairs,
                    depot_removals=removals,
                    next_station=next_station,
                    station_id=station_id
                )
                
                # Validate action feasibility
                try:
                    PostDecisionState.apply_action(state, action)
                    feasible_actions.append(action)
                except ValueError:
                    # Action violates constraints
                    continue
        
        return feasible_actions
    
    def _sample_micro_actions(self, max_pickup: int, max_delivery: int,
                             max_repairs: int, max_removals: int,
                             max_samples: int = 20) -> List[Tuple[int, int, int]]:
        """
        Sample micro-action space (rebalancing, repairs, removals).
        
        Uses importance sampling: focus on extreme values and zero.
        
        Returns:
            List of (rebalancing, repairs, removals) tuples
        """
        actions = set()
        
        # Always include do-nothing
        actions.add((0, 0, 0))
        
        # Add boundary points
        if max_pickup > 0:
            actions.add((-max_pickup, 0, 0))  # Max pickup
        if max_delivery > 0:
            actions.add((max_delivery, 0, 0))  # Max delivery
        if max_repairs > 0:
            actions.add((0, max_repairs, 0))  # Max repairs
        if max_removals > 0:
            actions.add((0, 0, max_removals))  # Max removals
        
        # Sample combinations
        while len(actions) < max_samples:
            # Sample rebalancing
            if max_pickup + max_delivery > 0:
                rebal = self.rng.integers(-max_pickup, max_delivery + 1)
            else:
                rebal = 0
            
            # Sample repairs
            repairs = self.rng.integers(0, max_repairs + 1) if max_repairs > 0 else 0
            
            # Sample removals (constrained by remaining capacity after rebalancing)
            capacity_after_rebal = max_removals
            if rebal < 0:  # Pickup consumes capacity
                capacity_after_rebal = max(0, max_removals + rebal)
            removals = self.rng.integers(0, capacity_after_rebal + 1) if capacity_after_rebal > 0 else 0
            
            actions.add((rebal, repairs, removals))
        
        return list(actions)
    
    def _get_candidate_routing_stations(self, simul, current_station_id: str,
                                       max_candidates: int = 10) -> List[str]:
        """
        Get candidate stations for routing.
        
        Uses spatial proximity and demand patterns.
        
        Returns:
            List of station IDs to consider
        """
        # For now, use simple approach: nearest neighbors
        # In production, would use demand forecasts and value estimates
        
        candidates = [current_station_id]  # Can stay at current station
        
        # Add all other stations (simplified)
        # In practice, would filter by distance and demand
        # This would need access to the simulator state
        
        # Placeholder: return current station + a few random ones
        # TODO: Implement proper routing candidate selection
        
        return candidates[:max_candidates]
    
    def _mdp_action_to_simulator_action(self, mdp_action: MDPAction, 
                                       vehicle, simul) -> Action:
        """
        Convert MDP action to simulator Action object.
        
        Args:
            mdp_action: MDP action (ι, m_rep, m_rem, ρ)
            vehicle: Simulator vehicle
            simul: Simulator object
        
        Returns:
            Simulator Action object
        """
        current_station = vehicle.current_station
        
        # === Determine Bikes for Each Operation ===
        
        # 1. On-site repairs
        bikes_to_repair = self._select_bikes_for_repair(
            current_station, mdp_action.onsite_repairs
        )
        
        # 2. Depot removals (pickups)
        bikes_to_remove = self._select_bikes_for_removal(
            current_station, mdp_action.depot_removals
        )
        
        # 3. Rebalancing
        if mdp_action.rebalancing < 0:
            # Pickup functional bikes
            pickup_bikes = self._select_bikes_for_pickup(
                current_station, -mdp_action.rebalancing
            )
            delivery_bikes = []
        elif mdp_action.rebalancing > 0:
            # Deliver functional bikes
            pickup_bikes = []
            delivery_bikes = self._select_bikes_for_delivery(
                vehicle, mdp_action.rebalancing
            )
        else:
            pickup_bikes = []
            delivery_bikes = []
        
        # Combine pickups (depot removals + rebalancing pickups)
        all_pickups = bikes_to_remove + pickup_bikes
        
        # Calculate maintenance time
        maintenance_time = len(bikes_to_repair) * self.repair_time_per_bike
        
        # Create simulator action
        return Action(
            battery_swaps=[],  # Not used in this implementation
            pick_ups=all_pickups,
            delivery_bikes=delivery_bikes,
            next_location=mdp_action.next_station,
            maintenance_time=maintenance_time
        )
    
    def _select_bikes_for_repair(self, station, count: int) -> List:
        """Select bikes for on-site repair."""
        bikes = station.get_bikes()
        repair_bikes = [b for b in bikes if hasattr(b, 'damage_status') 
                       and b.damage_status == 'onsite']
        return repair_bikes[:count]
    
    def _select_bikes_for_removal(self, station, count: int) -> List:
        """Select bikes for depot removal."""
        bikes = station.get_bikes()
        depot_bikes = [b for b in bikes if hasattr(b, 'damage_status') 
                      and b.damage_status == 'depot']
        return depot_bikes[:count]
    
    def _select_bikes_for_pickup(self, station, count: int) -> List:
        """Select functional bikes for pickup."""
        bikes = station.get_available_bikes()  # Only functional bikes
        return bikes[:count]
    
    def _select_bikes_for_delivery(self, vehicle, count: int) -> List:
        """Select bikes from vehicle cargo for delivery."""
        if not hasattr(vehicle, 'bikes'):
            return []
        
        functional_bikes = [b for b in vehicle.bikes 
                           if not hasattr(b, 'damage_status') 
                           or b.damage_status is None]
        return functional_bikes[:count]
    
    def save_agent(self, filepath: Path):
        """Save learned VFA model."""
        self.vfa_agent.save_model(filepath)
    
    def load_agent(self, filepath: Path):
        """Load pre-trained VFA model."""
        self.vfa_agent.load_model(filepath)
    
    def set_learning_mode(self, mode: bool):
        """Enable/disable learning during simulation."""
        self.learning_mode = mode
        print(f"VFAPolicy learning mode set to: {mode}")
