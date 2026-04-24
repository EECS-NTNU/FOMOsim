from __future__ import annotations

import copy
import numpy as np
import sim
from typing import TYPE_CHECKING, Optional
from policies.policy import Policy
from policies.sjovik_sund.mdp.reward import RewardCalculator
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy

if TYPE_CHECKING:
    from policies.sjovik_sund.vfa.run_logger import RunLogger

class HybridRolloutPolicy(Policy):
    def __init__(
        self,
        trained_vfa: LinearVFAPolicy,
        lookahead_minutes: float = 60.0,
        num_scenarios: int = 5,
        logger: Optional[RunLogger] = None,
    ):
        super().__init__(maintenance_enabled=trained_vfa.maintenance_enabled)

        self.vfa = trained_vfa
        self.vfa.learning_mode = False  # STRICTLY NO LEARNING
        self.lookahead_minutes = lookahead_minutes
        self.num_scenarios = num_scenarios
        self._simulator = None
        self.weights = getattr(trained_vfa, "weights", [])
        self.N_ROLLOUT_CANDIDATES = 5  # Number of top candidates to fully simulate
        self.logger: Optional[RunLogger] = logger


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

        # Truncate the event queue to the rollout horizon.
        # The queue is a sorted list (bisect-inserted), but we use a filter rather
        # than break so this stays correct even if the ordering assumption ever changes.
        cutoff = rollout_sim.state.time + self.lookahead_minutes
        before = len(rollout_sim.event_queue)
        rollout_sim.event_queue = [e for e in rollout_sim.event_queue if e.time <= cutoff]
        print(f"[Rollout] Event queue truncated: {before} -> {len(rollout_sim.event_queue)} events (cutoff={cutoff:.1f} min)")

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
        best_rollout_reward = 0.0   # mean rollout reward (no tail) for chosen action
        best_tail_value = 0.0       # mean discounted tail value for chosen action

        # Pre-generate one seed pair per scenario so every candidate is evaluated
        # on the same stochastic futures (Common Random Numbers).
        crn_seeds = [
            (int(state.rng.integers(1_000_000)), int(state.rng.integers(1_000_000)))
            for _ in range(self.num_scenarios)
        ]

        # Temporarily mute VFA operational logging so rollouts don't corrupt your CSVs
        old_log_rl = getattr(self.vfa, 'log_rl_decisions', False)
        old_log_depot = getattr(self.vfa, 'log_depot_visits', False)
        self.vfa.log_rl_decisions = False
        self.vfa.log_depot_visits = False

        # 2. Evaluate each candidate via Lookahead
        for action in candidates:
            expected_q = 0.0
            expected_rollout_only = 0.0  # accumulated reward without tail
            expected_tail = 0.0          # discounted tail contribution

            for omega in range(self.num_scenarios):
                rollout_sim = self._clone_simulator()
                sim_state = rollout_sim.state

                # Pin this clone's RNGs to the shared scenario seed so all
                # candidates experience the same demand/travel-time realisations.
                rng_seed, rng2_seed = crn_seeds[omega]
                sim_state.rng  = np.random.default_rng(rng_seed)
                sim_state.rng2 = np.random.default_rng(rng2_seed)

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

                # Terminal Evaluation (The Tail Value)
                terminal_vehicle = sim_state.get_vehicle_by_id(vehicle.id)
                base_func, base_onsite, base_depot = self.vfa._extract_inventories(sim_state, terminal_vehicle)
                terminal_phi = self.vfa.extract_features(
                    sim_state, terminal_vehicle, base_func, base_onsite, base_depot
                )
                terminal_value = self.vfa.value(terminal_phi)
                terminal_discount = self.vfa.gamma ** (self.lookahead_minutes / 60.0)
                discounted_tail = terminal_discount * terminal_value

                expected_q            += (accumulated_reward + discounted_tail) / self.num_scenarios
                expected_rollout_only += accumulated_reward / self.num_scenarios
                expected_tail         += discounted_tail / self.num_scenarios

            if expected_q > best_q_value:
                best_q_value = expected_q
                best_action = action
                best_rollout_reward = expected_rollout_only
                best_tail_value = expected_tail

        # Restore VFA logging config for the real simulation step
        self.vfa.log_rl_decisions = old_log_rl
        self.vfa.log_depot_visits = old_log_depot

        if self.logger is not None and best_action is not None:
            self._log_decision(state, vehicle, best_action,
                               best_rollout_reward, best_tail_value, best_q_value)

        return best_action

    def _log_decision(self, state, vehicle, action,
                      rollout_reward: float, tail_value: float, final_score: float) -> None:
        """Build a decision row and forward it to the RunLogger."""
        logger = self.logger
        if logger is None:
            return
        current_time = state.time
        clock_min  = current_time % (24 * 60)
        day        = int(current_time // (24 * 60))
        clock_hour = int(clock_min // 60)
        minute     = int(clock_min % 60)

        # Vehicle inventory before action
        inv         = vehicle.get_bike_inventory()
        func_before  = sum(1 for b in inv if getattr(b, "damage_status", None) not in ("depot", "onsite"))
        depot_before = sum(1 for b in inv if getattr(b, "damage_status", None) == "depot")
        total_before = len(inv)

        # Classify pickups by damage status
        station_bikes    = getattr(vehicle.location, "bikes", {})
        func_pickups     = 0
        depot_pickups    = 0
        for b_id in getattr(action, "pick_ups", []):
            b = station_bikes.get(b_id)
            if b and getattr(b, "damage_status", None) == "depot":
                depot_pickups += 1
            else:
                func_pickups += 1

        func_deliveries = len(getattr(action, "delivery_bikes", []))
        onsite_repairs  = len(getattr(action, "onsite_repairs", []))
        is_at_depot     = vehicle.is_at_depot()
        # Broken bikes the vehicle brought to the depot — will be automatically unloaded
        depot_deliveries = depot_before if is_at_depot else 0
        load_from_queue  = int(getattr(action, "load_from_queue", 0))

        # Planned vehicle load after action
        if is_at_depot:
            func_after  = func_before - func_deliveries + load_from_queue
            depot_after = 0  # all broken bikes unloaded
        else:
            func_after  = func_before - func_deliveries + func_pickups
            depot_after = depot_before + depot_pickups
        total_after = max(func_after, 0) + max(depot_after, 0)

        # Travel time to next station
        dest = getattr(action, "next_location", None)
        try:
            travel_time = state.get_vehicle_travel_time(vehicle.location.id, dest) if dest else 0.0
        except Exception:
            travel_time = 0.0

        # Operation duration (station work only, not travel)
        try:
            action_duration = action.get_action_time(0.0) if hasattr(action, "get_action_time") else 0.0
        except Exception:
            action_duration = 0.0

        # Maintenance context flags
        n_damaged_at_station = sum(
            1 for b in station_bikes.values()
            if getattr(b, "damage_status", None) is not None
        )
        maintenance_flag_present    = (n_damaged_at_station > 0) or is_at_depot
        selected_action_is_maintenance = (onsite_repairs > 0) or (depot_pickups > 0) or is_at_depot

        # Bike IDs involved (pick-ups + deliveries)
        delivery_ids = [
            getattr(b, "bike_id", str(b)) for b in getattr(action, "delivery_bikes", [])
        ]
        bikes_involved = str(list(getattr(action, "pick_ups", [])) + delivery_ids)

        logger.log_decision({
            "day":    day,
            "hour":   clock_hour,
            "minute": minute,
            "current_station_id":       vehicle.location.id,
            "is_at_depot":              is_at_depot,
            "functional_load_before":   func_before,
            "depot_load_before":        depot_before,
            "total_load_before":        total_before,
            "functional_deliveries":    func_deliveries,
            "functional_pickups":       func_pickups,
            "onsite_repairs":           onsite_repairs,
            "depot_pickups":            depot_pickups,
            "depot_deliveries":         depot_deliveries,
            "load_from_queue":          load_from_queue,
            "bikes_involved":           bikes_involved,
            "action_duration_min":      round(action_duration, 2),
            "next_station_id":          str(dest) if dest else "",
            "travel_time_min":          round(travel_time, 2),
            "functional_load_after":    max(func_after, 0),
            "depot_load_after":         max(depot_after, 0),
            "total_load_after":         total_after,
            "immediate_reward":         0.0,
            "accumulated_rollout_reward": round(rollout_reward, 6),
            "tail_value":               round(tail_value, 6),
            "final_decision_score":     round(final_score, 6),
            "maintenance_flag_present": maintenance_flag_present,
            "selected_action_is_maintenance": selected_action_is_maintenance,
        })
