from dataclasses import dataclass
from typing import Optional

@dataclass
class RewardConfig:
    # --- Operational Components ---
    weight_starvation: float = -1.0
    weight_congestion: float = -1.0
    weight_maintenance_violation: float = 0.0  # Set to >0 to penalize broken bikes left alone
    
    # --- End-of-Day Components (from your existing code) ---
    not_at_depot_at_end_penalty: float = 0.0 #-1000.0
    functional_bikes_at_end_penalty: float = 0.0 #-50.0
    
    @staticmethod
    def benchmark_base_only() -> "RewardConfig":
        """Configuration for isolating just starvation and congestion."""
        return RewardConfig(
            weight_starvation=-1.0,
            weight_congestion=-1.0,
            not_at_depot_at_end_penalty=0.0,
            functional_bikes_at_end_penalty=0.0
        )

class RewardCalculator:
    def __init__(self, config: Optional[RewardConfig] = None, gamma: float = 0.99):
        self.config = config or RewardConfig()
        self.gamma = gamma
        self._scale_factor = 1.0 - gamma  # Will be 0.01 when gamma=0.99
       
        # Move the state tracking out of the policy and into the calculator
        self._prev_starvations = 0
        self._prev_congestions = 0
        self._prev_trips = 0  # <--- NEW: Track trips for exact service level'''

        # --- FIXED: Set to 1.0. Stop crushing the reward signal! ---
        #self._scale_factor = 1.0  - self.gamma  # This will be 0.01 when gamma=0.99
        #self._scale_factor = 1.0 # Have removed scaling due to it cerushing the reward signal and causing instability. The gamma discounting will still be applied during learning updates, so we can afford to keep the raw reward values intact for better learning dynamics.
        # Articles reagarding this:
        # https://www.nature.com/articles/nature14236 but it is deep RL, but they seem to do discouted rewards with gamma, but no immediate reward dampening?
        # 
       
        # Move the state tracking out of the policy and into the calculator
        self._prev_starvations = 0
        self._prev_congestions = 0
        self._prev_trips = 0

    def reset_episode(self):
        """Must be called at the start of every 14-day episode."""
        self._prev_starvations = 0
        self._prev_congestions = 0
        self._prev_trips = 0  # <--- NEW

    def compute_step_reward(self, simulator_metrics) -> float:
        """Calculates the reward since the last decision epoch."""
        cur_s = simulator_metrics.get_aggregate_value("starvations") or 0
        cur_c = simulator_metrics.get_aggregate_value("long congestions") or 0
        cur_t = simulator_metrics.get_aggregate_value("trips") or 0  # <--- NEW

        delta_s = cur_s - self._prev_starvations
        delta_c = cur_c - self._prev_congestions

        reward = 0.0
        reward += self.config.weight_starvation * delta_s
        reward += self.config.weight_congestion * delta_c

        # Update trackers for the next step
        self._prev_starvations = cur_s
        self._prev_congestions = cur_c
        self._prev_trips = cur_t  # <--- NEW: Store the trips snapshot
 
        return reward * self._scale_factor