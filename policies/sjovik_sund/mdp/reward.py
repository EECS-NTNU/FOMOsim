from dataclasses import dataclass
from typing import Optional

@dataclass
class RewardConfig:
    # --- Operational Components ---
    weight_starvation: float = -1.0
    weight_congestion: float = -1.0
    weight_trip_served: float = 0.0           # Positive reward per successful trip (trips - starv - cong)
    weight_maintenance_violation: float = 0.0  # Set to >0 to penalize broken bikes left alone
    weight_fleet_degradation: float = 0.0     # Penalizes total_broken / total_fleet ratio each step
    use_reward_centering: bool = False
    reward_centering_beta: float = 0.01

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

    @staticmethod
    def benchmark_with_maintenance() -> "RewardConfig":
        """Starvation + congestion + fleet degradation penalty (active maintenance signal)."""
        return RewardConfig(
            weight_starvation=-1.0,
            weight_congestion=-1.0,
            weight_fleet_degradation=-0.0,
            not_at_depot_at_end_penalty=0.0,
            functional_bikes_at_end_penalty=0.0,
        )

    @staticmethod
    def benchmark_with_shift_penalty() -> "RewardConfig":
        """Starvation + congestion + late-shift depot-return penalty.
        Penalizes the vehicle for being away from depot with cargo in the last 2 hours of shift.
        """
        return RewardConfig(
            weight_starvation=-1.0,
            weight_congestion=-1.0,
            not_at_depot_at_end_penalty=-2.0,
            functional_bikes_at_end_penalty=-0.2,
        )

class RewardCalculator:
    def __init__(self, config: Optional[RewardConfig] = None, gamma: float = 0.99):
        self.config = config or RewardConfig()
        self.gamma = gamma
        #self._scale_factor = 1.0 - gamma  # Will be 0.01 when gamma=0.99

        # --- FIXED: Set to 1.0. Stop crushing the reward signal! ---
        #self._scale_factor = 1.0  - self.gamma  # This will be 0.01 when gamma=0.99
        self._scale_factor = 0.1 # Have removed scaling due to it cerushing the reward signal and causing instability. The gamma discounting will still be applied during learning updates, so we can afford to keep the raw reward values intact for better learning dynamics.
        # Articles reagarding this:
        # https://www.nature.com/articles/nature14236 but it is deep RL, but they seem to do discouted rewards with gamma, but no immediate reward dampening?
        # 
       
        # Move the state tracking out of the policy and into the calculator
        self._prev_starvations = 0
        self._prev_congestions = 0
        self._prev_trips = 0
        self._running_reward_mean = 0.0
        self._reward_mean_initialized = False

    def reset_episode(self):
        """Must be called at the start of every 14-day episode."""
        self._prev_starvations = 0
        self._prev_congestions = 0
        self._prev_trips = 0  # <--- NEW

    def center_reward(self, reward: float) -> float:
        """Subtract a running reward baseline for TD stability when enabled."""
        if not self.config.use_reward_centering:
            return reward

        beta = min(max(float(self.config.reward_centering_beta), 0.0), 1.0)
        if not self._reward_mean_initialized:
            self._running_reward_mean = reward
            self._reward_mean_initialized = True
            return 0.0

        centered = reward - self._running_reward_mean
        self._running_reward_mean = (
            (1.0 - beta) * self._running_reward_mean
            + beta * reward
        )
        return centered

    def compute_fleet_penalty(self, sim_state) -> float:
        """
        Penalize total_broken / total_fleet ratio at this moment.
        Call once per decision epoch alongside compute_step_reward.
        Returns a negative float (or 0.0 if weight is 0 or no stations).
        """
        if self.config.weight_fleet_degradation == 0.0:
            return 0.0
        total_bikes  = 0
        total_broken = 0
        for s in sim_state.get_stations():
            for bike in s.get_bikes():
                total_bikes += 1
                ds = getattr(bike, "damage_status", None)
                if ds in ("depot", "onsite"):
                    total_broken += 1
        if total_bikes == 0:
            return 0.0
        ratio = total_broken / total_bikes
        return self.config.weight_fleet_degradation * ratio * self._scale_factor

    def compute_late_shift_penalty(self, vehicle, state) -> float:
        """Continuous penalty in final 2 hours of shift when vehicle is away from depot with cargo.
        Ramps linearly from 0 at 120 min remaining to full penalty at shift end.
        Enable by setting not_at_depot_at_end_penalty and/or functional_bikes_at_end_penalty != 0.
        """
        if (self.config.not_at_depot_at_end_penalty == 0.0 and
                self.config.functional_bikes_at_end_penalty == 0.0):
            return 0.0
        if vehicle.is_at_depot():
            return 0.0
        from settings import SERVICE_TIME_TO  # avoid circular import at module level
        _close_min = SERVICE_TIME_TO * 60.0
        _clock_min = state.time % 1440.0
        time_remaining = max(0.0, _close_min - _clock_min)
        _penalty_window = 120.0  # minutes from shift end where penalty activates
        if time_remaining >= _penalty_window:
            return 0.0
        urgency = 1.0 - (time_remaining / _penalty_window)  # 0 at 120 min, 1 at shift end
        penalty = self.config.not_at_depot_at_end_penalty * urgency
        if self.config.functional_bikes_at_end_penalty != 0.0:
            n_func = sum(
                1 for b in vehicle.get_bike_inventory()
                if getattr(b, "damage_status", None) not in ("depot", "onsite")
            )
            penalty += self.config.functional_bikes_at_end_penalty * n_func * urgency
        return penalty * self._scale_factor

    def compute_step_reward(self, simulator_metrics) -> float:
        """Calculates the reward since the last decision epoch."""
        cur_s = simulator_metrics.get_aggregate_value("starvations") or 0
        cur_c = simulator_metrics.get_aggregate_value("long congestions") or 0
        cur_t = simulator_metrics.get_aggregate_value("trips") or 0  # <--- NEW

        delta_s = cur_s - self._prev_starvations
        delta_c = cur_c - self._prev_congestions
        delta_t = cur_t - self._prev_trips

        reward = 0.0
        reward += self.config.weight_starvation * delta_s
        reward += self.config.weight_congestion * delta_c
        if self.config.weight_trip_served != 0.0:
            served = max(0, delta_t - delta_s - delta_c)
            reward += self.config.weight_trip_served * served

        # Update trackers for the next step
        self._prev_starvations = cur_s
        self._prev_congestions = cur_c
        self._prev_trips = cur_t

        return reward * self._scale_factor
