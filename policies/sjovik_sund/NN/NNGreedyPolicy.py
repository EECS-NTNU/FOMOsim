'''import torch

from policies.policy import Policy
from policies.sjovik_sund.mdp.mdp_formulation import extract_mdp_state, PostDecisionState
from policies.sjovik_sund.mdp.mdp_config import MDPConfig
from policies.sjovik_sund.mdp.candidate_generator_nn import generate_candidates
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

    def _mdp_action_allowed(self, action) -> bool:
        if not self.config.allow_onsite_repairs and action.onsite_repairs != 0:
            return False
        if not self.config.allow_depot_removals:
            if action.depot_removals != 0 or action.depot_dropoffs != 0 or action.load_from_queue != 0:
                return False
        return True

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
                if not self._mdp_action_allowed(mdp_action):
                    continue
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
from policies.sjovik_sund.mdp.candidate_generator_nn import generate_candidates
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

    def _mdp_action_allowed(self, action) -> bool:
        if not self.config.allow_onsite_repairs and action.onsite_repairs != 0:
            return False
        if not self.config.allow_depot_removals:
            if action.depot_removals != 0 or action.depot_dropoffs != 0 or action.load_from_queue != 0:
                return False
        return True

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
                if not self._mdp_action_allowed(mdp_action):
                    continue
                try:
                    post_state, action_duration, _ = PostDecisionState.apply(mdp_state, mdp_action)
                    dest = mdp_action.next_station
                    dest_tt = {
                        sid: state.get_vehicle_travel_time(dest, sid)
                        for sid in mdp_state.stations
                    }
                    enc = encode_state(
                        post_state,
                        dest_travel_times=dest_tt,
                        mdp_action=mdp_action,
                        action_duration=action_duration,
                    )
                except Exception:
                    continue
                
                v = self.nn_model(
                    enc["station_block"].to(self._device),
                    enc["vehicle_block"].to(self._device),
                    enc["global_context"].to(self._device),
                ).item()
                
                if v > best_value:
                    best_value  = v
                    best_action = sim_action

        return best_action
