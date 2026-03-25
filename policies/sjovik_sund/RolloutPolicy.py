"""
RolloutPolicy.py  –  Horizontal Rollout Algorithm with VFA Base Policy & Tail Value

Combines a trained LinearVFAPolicy as:
  1. Base Policy: Drives decisions during lookahead (greedy, tau ≈ 0)
  2. Tail Value Function: Evaluates state at horizon H

For each real decision, evaluates candidates via N stochastic scenarios over H epochs.

Usage:
    from policies.sjovik_sund.RolloutPolicy import RolloutPolicy
    
    vfa_policy = LinearVFAPolicy.load("path/to/vfa.pkl")
    vfa_policy.tau = 0.001
    vfa_policy.learning_mode = False
    
    rollout = RolloutPolicy(
        vfa_policy=vfa_policy,
        reward_calculator=RewardCalculator(),
        num_scenarios=50,
        horizon=4,
    )
    
    simulator = run_simulation(policy=rollout, ...)
"""

from __future__ import annotations

import sys
import copy
import traceback
import numpy as np
from pathlib import Path
from typing import List, Optional

WORKSPACE_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.policy import Policy
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.mdp.reward import RewardCalculator
from policies.sjovik_sund.mdp.action_bridge import mdp_action_to_sim_action
from policies.sjovik_sund.mdp.mdp_formulation import MdpAction
import sim


class RolloutPolicy(Policy):
    """
    Horizontal rollout algorithm combining VFA as base policy and tail value function.
    
    Scores each candidate action by simulating N stochastic clones over H decision epochs,
    accumulating discounted rewards, and extracting tail value at the horizon.
    
    Parameters:
        vfa_policy (LinearVFAPolicy): Trained VFA (frozen weights, greedy mode).
        reward_calculator (RewardCalculator): Computes step rewards during rollout.
        num_scenarios (int): Number of stochastic clones per candidate (default 50).
        horizon (int): Lookahead horizon in decisions (default 4).
        use_vfa_candidates (bool): If True, use VFA candidate generator.
        seed (int): Random seed (default 42).
    """

    def __init__(
        self,
        vfa_policy: LinearVFAPolicy,
        reward_calculator: RewardCalculator,
        num_scenarios: int = 50,
        horizon: int = 4,
        use_vfa_candidates: bool = True,
        seed: int = 1042,
        source_simulator = None,
    ) -> None:
        super().__init__(maintenance_enabled=vfa_policy.maintenance_enabled)
        
        self.vfa_policy = vfa_policy
        self.reward_calc = reward_calculator
        self.num_scenarios = num_scenarios
        self.horizon = horizon
        self.use_vfa_candidates = use_vfa_candidates
        self._rng = np.random.default_rng(seed)
        self.source_simulator = source_simulator  # Reference to main simulator for creating clones
        
        # Ensure greedy mode
        self.vfa_policy.learning_mode = False
        self.vfa_policy.tau = max(self.vfa_policy.tau, 1e-6)
        
        # For logging
        self.weights = self.vfa_policy.weights

    def get_best_action(self, state: sim.State, vehicle: sim.Vehicle) -> sim.Action:
        """
        Execute the horizontal rollout algorithm.
        Evaluates candidates and returns highest-scoring action.
        """
        
        # Phase 1: Generate Candidates
        candidates = self.vfa_policy._generate_candidates(state, vehicle)
        
        if not candidates:
            print(f"[ROLLOUT] No candidates generated for Vehicle {vehicle.id}, using fallback action.")
            return self._fallback_action(state, vehicle)
        
        print(f"[ROLLOUT] Vehicle {vehicle.id} | Evaluating {len(candidates)} candidates "
              f"with {self.num_scenarios} scenarios x {self.horizon} horizon")
        
        # Score each candidate across N scenarios
        candidate_scores: List[float] = []
        
        for candidate_idx, candidate_action in enumerate(candidates):
            scores_across_scenarios = []
            
            for scenario in range(self.num_scenarios):
                score = self._rollout_single_scenario(
                    state, vehicle, candidate_action, scenario
                )
                scores_across_scenarios.append(score)
            
            avg_score = float(np.mean(scores_across_scenarios))
            candidate_scores.append(avg_score)
            std_score = float(np.std(scores_across_scenarios)) if len(scores_across_scenarios) > 1 else 0.0
            
            print(f"  Candidate {candidate_idx+1}/{len(candidates)}: "
                  f"avg_score={avg_score:.4f}, std={std_score:.4f}")
        
        # Pick the best candidate
        best_idx = int(np.argmax(candidate_scores))
        best_action = candidates[best_idx]
        best_score = candidate_scores[best_idx]
        
        print(f"[ROLLOUT] Selected candidate {best_idx+1}: score={best_score:.4f}\n")
        
        return best_action

    def _rollout_single_scenario(
        self,
        state: sim.State,      # Kept for signature compatibility
        vehicle: sim.Vehicle,
        candidate_action: sim.Action,
        scenario_id: int,
    ) -> float:
        """
        Simulate one scenario: execute candidate, rollout H-1 steps, compute tail value.
        """
        if not self.source_simulator:
            raise ValueError("RolloutPolicy requires a reference to the main source_simulator to clone.")

        # 1. Create a safe, fast clone of the entire simulation state and event queue
        temp_sim = self.source_simulator.sloppycopy()

        # Update the lookahead horizon time limit
        lookahead_duration = self.horizon * 3600  
        temp_sim.end_time = temp_sim.state.time + lookahead_duration

        # Optional: Suppress logging on the clone so it doesn't write to output files
        temp_sim.log_bike_movement = lambda *args, **kwargs: None
        temp_sim.log_trip_request = lambda *args, **kwargs: None
        temp_sim.log_component_failure = lambda *args, **kwargs: None

        # Seed the clone for stochastic scenarios
        scenario_seed = self._rng.integers(0, 2**31 - 1) + scenario_id
        temp_sim.state.set_seed(scenario_seed)

        # 2. Get the cloned vehicle reference
        cloned_vehicle = temp_sim.state.vehicles.get(vehicle.id)
        if cloned_vehicle is None:
            return 0.0

        # Mute other vehicles or enforce the base VFA policy on them
        for vid, v in temp_sim.state.vehicles.items():
            v.policy = self.vfa_policy

        cumulative_reward = 0.0
        discount_factor = 1.0

        self._sync_reward_calculator_baseline(temp_sim.state)

        # 3. Apply the Candidate Action
        self._apply_action_to_vehicle(cloned_vehicle, candidate_action, temp_sim.state)
        
        # Advance the simulation
        immediate_reward = self._advance_until_vehicle_ready(temp_sim, cloned_vehicle)
        cumulative_reward += discount_factor * immediate_reward

        # 4. Rollout remaining H-1 decisions using the base VFA policy
        for step in range(1, self.horizon):
            discount_factor *= self.vfa_policy.gamma
            self._sync_reward_calculator_baseline(temp_sim.state)
            
            # Fetch the next action using VFA iteratively
            next_action = self.vfa_policy.get_best_action(temp_sim.state, cloned_vehicle)
            self._apply_action_to_vehicle(cloned_vehicle, next_action, temp_sim.state)
            
            step_reward = self._advance_until_vehicle_ready(temp_sim, cloned_vehicle)
            cumulative_reward += discount_factor * step_reward

        # 5. Extract tail value (Cost-to-go) at horizon
        discount_factor *= self.vfa_policy.gamma
        tail_value = self._extract_tail_value(temp_sim.state, cloned_vehicle)
        cumulative_reward += discount_factor * tail_value
        
        return cumulative_reward

    def _lightweight_clone_state(self, state: sim.State) -> sim.State:
        """Safely isolate the physical inventory without deepcopying everything."""
        cloned_state = copy.copy(state)
        
        # 1. Isolate Stations AND their bike lists
        cloned_state.stations = {}
        for sid, real_station in state.stations.items():
            ghost_station = copy.copy(real_station)
            if isinstance(real_station.bikes, dict):
                ghost_station.bikes = {bid: copy.copy(b) for bid, b in real_station.bikes.items()}
            else:
                ghost_station.bikes = [copy.copy(b) for b in real_station.bikes]
            cloned_state.stations[sid] = ghost_station
            
        # 2. Isolate Vehicles AND their cargo
        cloned_state.vehicles = {}
        for vid, real_vehicle in state.vehicles.items():
            ghost_vehicle = copy.copy(real_vehicle)
            
            # --- THE FIX: Reconstruct the dictionary properly ---
            if isinstance(real_vehicle.bike_inventory, dict):
                ghost_vehicle.bike_inventory = {
                    bid: copy.copy(b) for bid, b in real_vehicle.bike_inventory.items()
                }
            else:
                # Fallback just in case it actually was a list
                ghost_vehicle.bike_inventory = [copy.copy(b) for b in real_vehicle.bike_inventory]
            # --------------------------------------------------
            
            # Re-link the vehicle's location to the new GHOST station
            if real_vehicle.location and hasattr(real_vehicle.location, 'id'):
                if real_vehicle.location.id in cloned_state.stations:
                    ghost_vehicle.location = cloned_state.stations[real_vehicle.location.id]
            cloned_state.vehicles[vid] = ghost_vehicle
            
        # 3. Isolate Metrics
        cloned_state.metrics = copy.deepcopy(state.metrics)
            
        return cloned_state

    def _sync_reward_calculator_baseline(self, cloned_state: sim.State) -> None:
        """Sync calculator baseline to current metric state."""
        self.reward_calc._prev_starvations = cloned_state.metrics.get_aggregate_value("starvations") or 0
        self.reward_calc._prev_congestions = cloned_state.metrics.get_aggregate_value("long congestions") or 0

    def _apply_action_to_vehicle(
        self,
        vehicle: sim.Vehicle,
        action: sim.Action,
        state: sim.State,
    ) -> None:
        """Apply action to cloned vehicle."""
        next_location = getattr(action, 'next_location', None) \
                     or getattr(action, 'next_station', None)
        
        if next_location is None:
            return
        
        vehicle.next_location = next_location
        
        if isinstance(next_location, str) and next_location in state.locations:
            dest_location = state.locations[next_location]
        else:
            dest_location = next_location
        
        if dest_location and hasattr(state, 'get_travel_time'):
            travel_time = state.get_travel_time(vehicle.location.id, dest_location.id)
            vehicle.eta = travel_time + vehicle.handling_time + vehicle.parking_time
        else:
            vehicle.eta = vehicle.handling_time + vehicle.parking_time
        
        vehicle._pending_action = action

    def _create_lookahead_simulator(self, state: sim.State):
        import demand
        from sim.Simulator import Simulator
        
        d = demand.Demand()
        lookahead_duration = self.horizon * 3600  
        
        temp_sim = Simulator(
            initial_state=state,
            target_state=getattr(self.vfa_policy, 'target_state', None),
            demand=d,
            start_time=state.time,
            duration=lookahead_duration,
            verbose=False
        )
        
        # --- THE FIX: MOCK ALL LOGGING METHODS ---
        # Prevent the temp simulator from crashing BikeDeparture.py
        temp_sim.log_bike_movement = lambda *args, **kwargs: None
        temp_sim.log_trip_request = lambda *args, **kwargs: None
        temp_sim.log_component_failure = lambda *args, **kwargs: None
        
        return temp_sim

    def _advance_until_vehicle_ready(
        self,
        temp_sim,
        vehicle: sim.Vehicle,
    ) -> float:
        """
        Advance simulator until vehicle is ready (arrived at next_location).
        Uses a reusable simulator instance passed in (not created here).
        
        Includes robust error handling for edge cases during lookahead (e.g., BikeDeparture
        failures when insufficient bikes in neighboring stations).
        """
        
        cumulative_reward = 0.0
        initial_eta = vehicle.eta
        vehicle_arrival_target_time = temp_sim.state.time + initial_eta
        
        try:
            # Adjust VehicleArrival event timing if needed
            if initial_eta > 0:
                for event in temp_sim.event_queue:
                    if hasattr(event, 'vehicle') and event.vehicle.id == vehicle.id:
                        if type(event).__name__ == 'VehicleArrival':
                            event.time = vehicle_arrival_target_time
                            break
                temp_sim.event_queue.sort(key=lambda e: e.time)
            
            # Process events until vehicle arrives
            max_events = 1000  # Safety limit to prevent infinite loops during lookahead
            event_count = 0
            
            while temp_sim.state.time < temp_sim.end_time and event_count < max_events:
                # Check arrival condition
                if vehicle.eta <= 0 and vehicle.location.id == vehicle.next_location:
                    break
                
                if not temp_sim.event_queue:
                    break
                
                # Peek at next event
                next_event = temp_sim.event_queue[0]
                
                # Early exit if next event is beyond our arrival target
                if next_event.time > vehicle_arrival_target_time + 60 and vehicle.eta <= 0:
                    break
                
                try:
                    # Execute one event with error resilience
                    temp_sim.single_step()
                    
                    # Compute reward delta
                    step_reward = self.reward_calc.compute_step_reward(temp_sim.state.metrics)
                    cumulative_reward += step_reward
                    
                except (IndexError, KeyError, ValueError) as e:
                    # During lookahead, some events may fail due to incomplete cloned state
                    # (e.g., BikeDeparture when no bikes available in neighbors)
                    # Skip the problematic event and continue
                    event_type = type(next_event).__name__
                    print(f"[ROLLOUT] Event failed during lookahead (skipping): {event_type} - {type(e).__name__}")
                    
                    # Remove the problematic event and continue
                    if temp_sim.event_queue:
                        temp_sim.event_queue.pop(0)
                    
                except Exception as e:
                    # Unexpected error - log and bail out gracefully
                    print(f"[ROLLOUT] Unexpected error during lookahead: {e}")
                    break
                
                event_count += 1
            
            if event_count >= max_events:
                print(f"[ROLLOUT] Max events reached during lookahead (safety limit)")
                
        except Exception as e:
            print(f"[ROLLOUT] Critical error during vm advance: {e}")
            traceback.print_exc()
            return 0.0

        return cumulative_reward

    def _extract_tail_value(
        self,
        state: sim.State,
        vehicle: sim.Vehicle,
    ) -> float:
        """Extract tail value V(S^x_H) = θᵀ φ(S^x_H)."""
        func, onsite, depot = self.vfa_policy._extract_inventories(state, vehicle)
        phi = self.vfa_policy.extract_features(state, vehicle, func, onsite, depot)
        tail_value = self.vfa_policy.value(phi)
        return float(tail_value)

    def _fallback_action(
        self,
        state: sim.State,
        vehicle: sim.Vehicle,
    ) -> sim.Action:
        """Fallback when no candidates generated."""
        cur_id = vehicle.location.id
        all_stations = list(state.get_stations())
        
        if not all_stations:
            return sim.Action([], [], [], cur_id)
        
        nearest = min(
            all_stations,
            key=lambda s: state.get_travel_time(cur_id, s.id)
        )
        
        mdp_action = MdpAction(
            current_station=cur_id,
            rebalancing=0,
            onsite_repairs=0,
            depot_removals=0,
            load_from_queue=0,
            next_station=nearest.id,
        )
        
        return mdp_action_to_sim_action(mdp_action, state, vehicle)
