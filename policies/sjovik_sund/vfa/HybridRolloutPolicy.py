import copy
import numpy as np
import sim
from policies.policy import Policy
from policies.sjovik_sund.mdp.reward import RewardCalculator
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy

class HybridRolloutPolicy(Policy):
    def __init__(
        self, 
        trained_vfa: LinearVFAPolicy, 
        lookahead_minutes: float = 60.0,
        num_scenarios: int = 1 
    ):
        super().__init__(maintenance_enabled=trained_vfa.maintenance_enabled)
        
        self.vfa = trained_vfa
        self.vfa.learning_mode = False # STRICTLY NO LEARNING
        self.lookahead_minutes = lookahead_minutes
        self.num_scenarios = num_scenarios
        self._simulator = None
        self.weights = getattr(trained_vfa, "weights", [])
        self.N_ROLLOUT_CANDIDATES = 5  # Number of top candidates to fully simulate


    def __deepcopy__(self, memo):
        return self

    def init_sim(self, simulator):
        self._simulator = simulator
        self.vfa.init_sim(simulator)

    def _clone_simulator(self):
        if self._simulator is None:
            raise RuntimeError("HybridRolloutPolicy.init_sim() must be called before rollout.")
        root_simulator = self._simulator
        rollout_sim = root_simulator.sloppycopy()

        # `Simulator.sloppycopy()` builds a new Simulator, whose init() calls
        # `vehicle.policy.init_sim(...)` on the cloned vehicles. Because the
        # cloned vehicles still point to this same HybridRolloutPolicy object,
        # that side effect would otherwise overwrite `self._simulator` and make
        # the next candidate clone branch from an already-modified rollout.
        self._simulator = root_simulator
        self.vfa.init_sim(root_simulator)

        # Rollout branches are internal scoring work, not real operations.
        if hasattr(rollout_sim, "operation_logger") and rollout_sim.operation_logger is not None:
            rollout_sim.operation_logger.enabled = False
        if hasattr(rollout_sim.state, "operation_logger") and rollout_sim.state.operation_logger is not None:
            rollout_sim.state.operation_logger.enabled = False

        return rollout_sim

    def _make_rollout_reward_calculator(self, simulator):
        reward_calc = RewardCalculator(
            config=copy.deepcopy(self.vfa.reward_calc.config),
            gamma=self.vfa.reward_calc.gamma,
        )
        # Sync the baseline so we only track future penalties
        reward_calc.compute_step_reward(simulator.state.metrics)
        return reward_calc

    def _apply_action_to_sim(self, simulator, sim_vehicle, action):
        """Bridging function: applies the root candidate action manually"""
        state = simulator.state
        origin_station_id = sim_vehicle.location.id
        current_time = state.time

        refill_time = state.do_action(action, sim_vehicle, current_time)
        travel_time = state.get_vehicle_travel_time(origin_station_id, action.next_location)
        
        if hasattr(action, "get_action_time"):
            action_time = action.get_action_time(travel_time) + refill_time
        else:
            action_time = travel_time + refill_time + getattr(action, "maintenance_time", 0.0)
            
        arrival_time = current_time + action_time

        simulator.add_event(sim.VehicleArrival(arrival_time, sim_vehicle))
        sim_vehicle.eta = arrival_time
        return 0.0

    def get_best_action(self, state, vehicle):
        if not self.vfa._initialized:
            self.vfa._lazy_init(state)

        # 1. Generate the immediate candidate actions
        all_candidates = self.vfa._generate_candidates(state, vehicle)
        
        # 1b. Pre-score using the frozen VFA and prune to N_ROLLOUT_CANDIDATES
        base_func, base_onsite, base_depot = self.vfa._extract_inventories(state, vehicle)
        candidate_scores = []
        for action in all_candidates:
            functional_pickups = 0
            depot_pickups = 0
            station_bikes = getattr(vehicle.location, "bikes", {})
            for b_id in action.pick_ups:
                b = station_bikes.get(b_id)
                if b and getattr(b, 'damage_status', None) == 'depot':
                    depot_pickups += 1
                elif b:
                    functional_pickups += 1
            
            delta_func = len(action.delivery_bikes) - functional_pickups
            delta_depot_cargo = depot_pickups
            delta_onsite_repairs = len(getattr(action, "onsite_repairs", []))
            
            if vehicle.is_at_depot():
                vehicle_depot_cargo = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) == 'depot')
                delta_depot_cargo = -vehicle_depot_cargo
            
            dest_id = getattr(action, "next_location", getattr(action, "next_station", None))
            
            phi = self.vfa.extract_features(
                state, vehicle, 
                base_func, base_onsite, base_depot, 
                delta_func, delta_depot_cargo,
                delta_onsite_repairs,
                next_station_id=dest_id
            )
            vfa_score = self.vfa.value(phi)
            candidate_scores.append((vfa_score, action))
            
        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        candidates = [item[1] for item in candidate_scores[:self.N_ROLLOUT_CANDIDATES]]

        best_action = None
        best_q_value = -float('inf')

        # Temporarily mute VFA operational logging so rollouts don't corrupt your CSVs
        old_log_rl = getattr(self.vfa, 'log_rl_decisions', False)
        old_log_depot = getattr(self.vfa, 'log_depot_visits', False)
        self.vfa.log_rl_decisions = False
        self.vfa.log_depot_visits = False

        # 2. Evaluate each candidate via Lookahead
        for action in candidates:
            expected_q = 0.0
            
            for omega in range(self.num_scenarios):
                rollout_sim = self._clone_simulator()
                sim_state = rollout_sim.state
                sim_vehicle = sim_state.get_vehicle_by_id(vehicle.id)
                reward_calc = self._make_rollout_reward_calculator(rollout_sim)
                
                # -------------------------------------------------------------
                # 🚨 CRITICAL FIX: PREVENT EXPONENTIAL RECURSION 🚨
                # Force cloned vehicles to use the pure VFA! This delegates the 
                # rest of the lookahead directly to the native simulator engine.
                # -------------------------------------------------------------
                for v in sim_state.get_vehicles():
                    v.policy = self.vfa
                
                # Apply the candidate action to bridge the gap
                self._apply_action_to_sim(rollout_sim, sim_vehicle, action)
                
                # Rollout Phase: Fast-forward natively
                target_time = sim_state.time + self.lookahead_minutes
                accumulated_reward = 0.0
                
                while rollout_sim.event_queue:
                    next_event = rollout_sim.event_queue[0]
                    if next_event.time > target_time:
                        sim_state.time = target_time
                        break
                        
                    rollout_sim.single_step()
                    
                    step_reward = reward_calc.compute_step_reward(sim_state.metrics)
                    time_elapsed = sim_state.time - state.time
                    discount = self.vfa.gamma ** max(time_elapsed / 60.0, 0.0)
                    accumulated_reward += discount * step_reward
                    
                trajectory_reward = accumulated_reward
                
                # Terminal Evaluation (The Tail Value)
                terminal_vehicle = sim_state.get_vehicle_by_id(vehicle.id)
                base_func, base_onsite, base_depot = self.vfa._extract_inventories(sim_state, terminal_vehicle)
                terminal_phi = self.vfa.extract_features(
                    sim_state, terminal_vehicle, base_func, base_onsite, base_depot
                )
                terminal_value = self.vfa.value(terminal_phi)
                
                terminal_discount = self.vfa.gamma ** (self.lookahead_minutes / 60.0)
                trajectory_reward += terminal_discount * terminal_value
                
                expected_q += trajectory_reward / self.num_scenarios

            if expected_q > best_q_value:
                best_q_value = expected_q
                best_action = action

        # Restore VFA logging config for the real simulation step
        self.vfa.log_rl_decisions = old_log_rl
        self.vfa.log_depot_visits = old_log_depot

        return best_action
