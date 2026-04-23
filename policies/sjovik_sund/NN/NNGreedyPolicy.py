'''import torch

from policies.policy import Policy
from policies.sjovik_sund.mdp.mdp_formulation import extract_mdp_state, PostDecisionState
from policies.sjovik_sund.mdp.mdp_config import MDPConfig
from policies.sjovik_sund.mdp.candidate_generator import generate_candidates
from policies.sjovik_sund.NN.nn_model import NNValueNetwork
from policies.sjovik_sund.NN.nn_state_encoder import encode_state


class NNGreedyPolicy(Policy):
    """
    Lightweight greedy policy using the NN value function — no rollout/cloning.
    Used as the base policy for other vehicles during NNRolloutPolicy fast-forward.
    Equivalent to LinearVFAPolicy being the base policy in HybridRolloutPolicy.
    """

    def __init__(self, nn_model: NNValueNetwork, config: MDPConfig, depot_id: str = None):
        super().__init__(maintenance_enabled=config.allow_onsite_repairs)
        self.nn_model = nn_model
        self.config   = config
        self.depot_id = depot_id
        self._device  = next(nn_model.parameters()).device

    def get_best_action(self, state, vehicle):
        mdp_state = extract_mdp_state(
            sim_state=state,
            active_vehicle_id=vehicle.id,
            config=self.config,
            depot_id=self.depot_id,
            shift_end_time=getattr(vehicle, "shift_end_time", None),
        )
        pairs = generate_candidates(
            state=state,
            vehicle=vehicle,
            maintenance_enabled=self.maintenance_enabled,
            return_pairs=True,
            wide_search=True,
        )
        if not pairs:
            return None

        best_action = None
        best_value  = -float("inf")
        with torch.no_grad():
            for mdp_action, sim_action in pairs:
                try:
                    post_state, _, _ = PostDecisionState.apply(mdp_state, mdp_action)
                    enc = encode_state(post_state)
                except Exception:
                    enc = encode_state(mdp_state)
                v = self.nn_model(
                    enc["station_block"].to(self._device),
                    enc["vehicle_block"].to(self._device),
                    enc["global_context"].to(self._device),
                ).item()
                if v > best_value:
                    best_value  = v
                    best_action = sim_action
        return best_action'''
        
import torch

from policies.policy import Policy
from policies.sjovik_sund.mdp.mdp_formulation import extract_mdp_state, PostDecisionState
from policies.sjovik_sund.mdp.mdp_config import MDPConfig
from policies.sjovik_sund.mdp.candidate_generator import generate_candidates
from policies.sjovik_sund.NN.nn_model import NNValueNetwork
from policies.sjovik_sund.NN.nn_state_encoder import encode_state

class NNGreedyPolicy(Policy):
    """
    Lightweight greedy policy using the NN value function — no rollout/cloning.
    Used as the base policy for other vehicles during NNRolloutPolicy fast-forward.
    Equivalent to LinearVFAPolicy being the base policy in HybridRolloutPolicy.
    """

    def __init__(self, nn_model: NNValueNetwork, config: MDPConfig, depot_id: str = None):
        super().__init__(maintenance_enabled=config.allow_onsite_repairs)
        self.nn_model = nn_model
        self.config   = config
        self.depot_id = depot_id
        self._device  = next(nn_model.parameters()).device

    def get_best_action(self, state, vehicle):
        mdp_state = extract_mdp_state(
            sim_state=state,
            active_vehicle_id=vehicle.id,
            config=self.config,
            depot_id=self.depot_id,
            shift_end_time=getattr(vehicle, "shift_end_time", None),
        )
        
        pairs = generate_candidates(
            state=state,
            vehicle=vehicle,
            maintenance_enabled=self.maintenance_enabled,
            return_pairs=True,
            wide_search=True,
        )
        if not pairs:
            return None

        best_action = None
        best_value  = -float("inf")
        with torch.no_grad():
            for mdp_action, sim_action in pairs:
                try:
                    post_state, _, _ = PostDecisionState.apply(mdp_state, mdp_action)
                    enc = encode_state(post_state)
                except Exception:
                    continue # <--- SKIP IT! Don't let the NN see impossible actions.
                
                v = self.nn_model(
                    enc["station_block"].to(self._device),
                    enc["vehicle_block"].to(self._device),
                    enc["global_context"].to(self._device),
                ).item()
                
                if v > best_value:
                    best_value  = v
                    best_action = sim_action

        return best_action