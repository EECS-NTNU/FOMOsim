"""
NNRolloutPolicy.py  —  Neural Network Rollout Policy

A parallel branch to HybridRolloutPolicy that replaces the linear VFA
terminal value estimator with the NNValueNetwork from nn_model.py.

─────────────────────────────────────────────────────────────────────────────
POSITION IN THE OVERALL ARCHITECTURE
─────────────────────────────────────────────────────────────────────────────

The rollout algorithm evaluates candidate actions by:
  1. Cloning the simulator
  2. Applying the candidate action
  3. Fast-forwarding H minutes (lookahead horizon)
  4. Estimating the value of the resulting terminal state

In HybridRolloutPolicy, step 4 is:
    V_terminal = θᵀ φ(S^x_terminal)   (linear VFA)

In NNRolloutPolicy, step 4 is:
    V_terminal = NNValueNetwork(encode_state(S^x_terminal))

Everything else — candidate generation, simulator cloning, scenario
sampling, reward accumulation, action bridge — is structurally identical.

─────────────────────────────────────────────────────────────────────────────
WHAT IS DIFFERENT FROM HybridRolloutPolicy
─────────────────────────────────────────────────────────────────────────────

  HybridRolloutPolicy                   NNRolloutPolicy
  ─────────────────────────────────────────────────────────────────
  Requires a trained LinearVFAPolicy    Requires a trained NNValueNetwork
  Terminal value: θᵀ φ(S^x)            Terminal value: NN(encode_state(S^x))
  Candidate gen: via self.vfa           Candidate gen: generate_candidates()
    (the VFA does double duty as          from mdp/candidate_generator.py —
     both value estimator and             shared standalone function; no VFA
     candidate generator)                 dependency for this step
  Training: offline in train_vfa.py     Training: offline in train_nn_rollout.py
  ─────────────────────────────────────────────────────────────────

Note on candidate generation: proximity filtering, tabu coordination, and
maintenance enumeration live in mdp/candidate_generator.py and are shared
between the VFA and NN branches. The NN replaces the value function only.

─────────────────────────────────────────────────────────────────────────────
ROLLOUT FLOW  (per call to get_best_action)
─────────────────────────────────────────────────────────────────────────────

  get_best_action(state, vehicle)
      │
      ├─ _generate_candidates()            enumerate routing × maintenance options
      │
      └─ for each candidate action a:
             for each scenario ω (1..num_scenarios):
               │
               ├─ clone simulator
               ├─ assign VFA policy to cloned vehicles (prevent recursive rollout)
               ├─ apply candidate action a to clone
               │
               ├─ fast-forward clone for lookahead_minutes
               │       accumulate discounted step rewards (starvations/congestions)
               │
               ├─ extract terminal MDPState from clone
               ├─ encode terminal state  →  {station_block, vehicle_block, global}
               ├─ nn_model.forward(...)  →  V_terminal  (no gradient)
               │
               └─ Q(a, ω) = accumulated_reward + γ^H · V_terminal
             │
             Q(a) = mean over scenarios
         │
         select argmax_a Q(a)
         return corresponding sim.Action
"""

import copy
import torch
import sim

from policies.policy import Policy
from policies.sjovik_sund.mdp.mdp_formulation import extract_mdp_state
from policies.sjovik_sund.mdp.reward import RewardCalculator
from policies.sjovik_sund.mdp.candidate_generator import generate_candidates
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.NN.nn_model import NNValueNetwork
from policies.sjovik_sund.NN.nn_state_encoder import encode_state


class NNRolloutPolicy(Policy):
    """
    Rollout policy that uses a trained NNValueNetwork as the terminal value
    estimator.

    During inference (get_best_action), the NN is called with torch.no_grad()
    — no gradient tracking, no weight updates.  All learning happens offline
    in train_nn_rollout.py before this policy is instantiated.

    Args:
        nn_model              : trained NNValueNetwork (from train_nn_rollout or
                                loaded via torch.load).
        candidate_vfa         : a LinearVFAPolicy whose _generate_candidates()
                                method is borrowed for action enumeration.
                                Its VALUE WEIGHTS ARE NOT USED for scoring.
                                Set learning_mode=False before passing in.
        lookahead_minutes     : how far ahead to simulate per scenario (H).
        num_scenarios         : number of Monte Carlo rollout samples per candidate.
        maintenance_enabled   : whether depot/onsite maintenance actions are included.
        depot_id              : ID of the depot station, passed to extract_mdp_state.
    """

    def __init__(
        self,
        nn_model:           NNValueNetwork,
        candidate_vfa:      LinearVFAPolicy,
        lookahead_minutes:  float = 60.0,
        num_scenarios:      int   = 3,
        maintenance_enabled: bool = True,
        depot_id:           str   = None,
    ):
        super().__init__(maintenance_enabled=maintenance_enabled)

        # The NN used for terminal value estimation.
        # Set to eval mode: disables dropout/batchnorm randomness during inference.
        self.nn_model = nn_model
        self.nn_model.eval()

        # The VFA is used ONLY for _generate_candidates() and reward_calc config.
        # Its scoring weights are never queried; learning_mode must be False.
        self._vfa_for_candidates = candidate_vfa
        self._vfa_for_candidates.learning_mode = False

        self.lookahead_minutes   = lookahead_minutes
        self.num_scenarios       = num_scenarios
        self.depot_id            = depot_id

        # Discount factor: reuse from the candidate VFA for consistency.
        self.gamma = getattr(candidate_vfa, "gamma", 0.99)

        # Will be set by init_sim() once the simulator is attached.
        self._simulator = None

    # ─────────────────────────────────────────────────────────────────────────
    # Simulator attachment
    # ─────────────────────────────────────────────────────────────────────────

    def __deepcopy__(self, memo):
        # Prevent deep-copying the policy when the simulator clones itself.
        # Cloned vehicles must NOT point to a cloned version of this policy;
        # they need to point back to the candidate VFA so the rollout terminates
        # after one level (no recursive lookahead on lookahead).
        return self

    def init_sim(self, simulator):
        """Called by the simulator after the event queue is initialized."""
        self._simulator = simulator
        self._vfa_for_candidates.init_sim(simulator)

    # ─────────────────────────────────────────────────────────────────────────
    # Simulator cloning helpers (mirrors HybridRolloutPolicy)
    # ─────────────────────────────────────────────────────────────────────────

    def _clone_simulator(self):
        """
        Create a disposable copy of the simulator for one rollout scenario.

        The root simulator reference is preserved here because sloppycopy()
        triggers init_sim() on the cloned vehicles, which would otherwise
        overwrite self._simulator with the clone.

        Logging is disabled in the clone: rollout branches are internal
        scoring work, not real operations, so they must not write to CSVs.
        """
        if self._simulator is None:
            raise RuntimeError("NNRolloutPolicy.init_sim() must be called before rollout.")

        root_sim  = self._simulator
        clone_sim = root_sim.sloppycopy()

        # Restore root reference that sloppycopy() may have clobbered
        self._simulator = root_sim
        self._vfa_for_candidates.init_sim(root_sim)

        # Silence operational logging in the clone
        for attr in ("operation_logger",):
            logger = getattr(clone_sim, attr, None) or getattr(clone_sim.state, attr, None)
            if logger is not None:
                logger.enabled = False

        return clone_sim

    def _make_reward_calculator(self, simulator) -> RewardCalculator:
        """
        Build a fresh RewardCalculator synced to the clone's current metrics.

        Syncing on creation means we measure reward RELATIVE to the state
        at the point the rollout branch starts — not accumulated from
        the beginning of the simulation.
        """
        reward_calc = RewardCalculator(
            config=copy.deepcopy(self._vfa_for_candidates.reward_calc.config),
            gamma=self.gamma,
        )
        reward_calc.compute_step_reward(simulator.state.metrics)
        return reward_calc

    def _apply_action_to_clone(self, clone_sim, clone_vehicle, action):
        """
        Apply the candidate action to the cloned simulator and schedule the
        next VehicleArrival event so the event loop can continue from there.

        This mirrors HybridRolloutPolicy._apply_action_to_sim().
        """
        state = clone_sim.state
        origin_id    = clone_vehicle.location.id
        current_time = state.time

        refill_time  = state.do_action(action, clone_vehicle, current_time)
        travel_time  = state.get_vehicle_travel_time(origin_id, action.next_location)

        if hasattr(action, "get_action_time"):
            action_time = action.get_action_time(travel_time) + refill_time
        else:
            action_time = travel_time + refill_time + getattr(action, "maintenance_time", 0.0)

        arrival_time = current_time + action_time
        clone_sim.add_event(sim.VehicleArrival(arrival_time, clone_vehicle))
        clone_vehicle.eta = arrival_time

    # ─────────────────────────────────────────────────────────────────────────
    # Terminal value estimation  (the NN's role)
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_terminal_value(self, clone_sim_state, vehicle_id: int) -> float:
        """
        Estimate V(S^x_terminal) using the neural network.

        This replaces the linear VFA call:
            θᵀ φ(S^x_terminal)
        with:
            NNValueNetwork(encode_state(S^x_terminal))

        Steps:
          1. Extract MDPState from the cloned simulator at the terminal time
          2. Encode it into {station_block, vehicle_block, global_context}
          3. Forward pass through the NN (no gradient)
          4. Return the scalar as a Python float

        Args:
            clone_sim_state : sim.State at the end of the rollout horizon
            vehicle_id      : ID of the vehicle making the original decision

        Returns:
            float — estimated future value (will be negative; penalties only)
        """
        # Step 1: snapshot the terminal simulator state as an MDPState
        terminal_mdp_state = extract_mdp_state(
            sim_state=clone_sim_state,
            active_vehicle_id=vehicle_id,
            config=self._vfa_for_candidates.config,
            depot_id=self.depot_id,
        )

        # Step 2: encode into tensors (no handcrafted features; raw ratios only)
        encoded = encode_state(terminal_mdp_state)
        
        # Dynamically check which device the model is currently on
        device = next(self.nn_model.parameters()).device

        # Step 3: forward pass — torch.no_grad() ensures no gradients are
        # accumulated, keeping inference fast and memory-efficient
        with torch.no_grad():
            value_tensor = self.nn_model(
                encoded["station_block"].to(device),
                encoded["vehicle_block"].to(device),
                encoded["global_context"].to(device),
            )

        # Step 4: unwrap to a Python scalar
        return value_tensor.item()

    # ─────────────────────────────────────────────────────────────────────────
    # Main decision method
    # ─────────────────────────────────────────────────────────────────────────


    def get_best_action(self, state, vehicle):
        """
        Select the best action for the arriving vehicle using rollout + NN.

        For each candidate action, num_scenarios independent rollouts are run.
        The Q-value of an action is the average over scenarios of:

            Q(a, ω) = Σ_t γ^t r_t  +  γ^H · V_NN(S^x_terminal)

        where r_t are step rewards (starvation/congestion penalties) and
        V_NN is the neural network's terminal value estimate.

        The action with the highest (least negative) Q-value is returned.
        """
        # --- Enumerate candidate actions ---
        # Calls the shared standalone function from mdp/candidate_generator.py.
        # No VFA initialisation needed; generate_candidates() reads directly
        # from sim.State and sim.Vehicle.
        candidates = generate_candidates(
            state=state,
            vehicle=vehicle,
            maintenance_enabled=self.maintenance_enabled,
        )

        # Mute VFA logging for the rollout phase — cloned vehicles still use
        # _vfa_for_candidates as their internal policy, so its debug prints
        # would otherwise flood output during scoring.
        old_log_rl    = getattr(self._vfa_for_candidates, "log_rl_decisions", False)
        old_log_depot = getattr(self._vfa_for_candidates, "log_depot_visits", False)
        self._vfa_for_candidates.log_rl_decisions = False
        self._vfa_for_candidates.log_depot_visits = False

        best_action  = None
        best_q_value = -float("inf")

        # --- Score each candidate via Monte Carlo rollout ---
        for action in candidates:
            total_q = 0.0

            for _omega in range(self.num_scenarios):

                # 1. Create a disposable simulator clone for this scenario
                clone_sim     = self._clone_simulator()
                clone_state   = clone_sim.state
                clone_vehicle = clone_state.get_vehicle_by_id(vehicle.id)
                reward_calc   = self._make_reward_calculator(clone_sim)

                # 2. CRITICAL: force cloned vehicles to use the candidate VFA,
                #    not this NNRolloutPolicy. This prevents infinite recursion
                #    (rollout calling rollout calling rollout...). The lookahead
                #    uses the VFA greedily for internal decisions; the NN only
                #    evaluates the terminal state at the end of the horizon.
                for v in clone_state.get_vehicles():
                    v.policy = self._vfa_for_candidates

                # 3. Apply the candidate action and register the next arrival event
                self._apply_action_to_clone(clone_sim, clone_vehicle, action)

                # 4. Fast-forward the clone until the lookahead horizon is reached,
                #    accumulating discounted step rewards along the way
                target_time        = clone_state.time + self.lookahead_minutes
                accumulated_reward = 0.0

                while clone_sim.event_queue:
                    next_event = clone_sim.event_queue[0]
                    if next_event.time > target_time:
                        clone_state.time = target_time
                        break
                    clone_sim.single_step()

                    step_reward   = reward_calc.compute_step_reward(clone_state.metrics)
                    time_elapsed  = clone_state.time - state.time
                    # Discount grows with elapsed time; each 60-minute unit = one γ step
                    discount      = self.gamma ** max(time_elapsed / 60.0, 0.0)
                    accumulated_reward += discount * step_reward

                # 5. Estimate terminal value using the NN (replaces the VFA call
                #    θᵀ φ(S^x_terminal) in HybridRolloutPolicy)
                terminal_value    = self._estimate_terminal_value(clone_state, vehicle.id)
                terminal_discount = self.gamma ** (self.lookahead_minutes / 60.0)

                scenario_q = accumulated_reward + terminal_discount * terminal_value
                total_q   += scenario_q

            # Average Q-value across scenarios
            mean_q = total_q / self.num_scenarios

            if mean_q > best_q_value:
                best_q_value = mean_q
                best_action  = action

        # Restore VFA logging settings for the real simulation step
        self._vfa_for_candidates.log_rl_decisions = old_log_rl
        self._vfa_for_candidates.log_depot_visits = old_log_depot

        return best_action
