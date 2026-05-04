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

    # --- End-of-Day Components (from your existing code) ---
    not_at_depot_at_end_penalty: float = 0.0 #-1000.0
    functional_bikes_at_end_penalty: float = 0.0 #-50.0

    # --- Experiment switch: maintenance shaping (+10/+5/+5) ---
    # Set False for tail-value experiments — keeps value surface consistent with base reward.
    use_maintenance_shaping: bool = True
    
    @staticmethod
    def benchmark_base_only() -> "RewardConfig":
        """Configuration for isolating just starvation and congestion."""
        return RewardConfig(
            weight_starvation=-1.0,
            weight_congestion=-1.0,
            weight_trip_served=0.0,
            not_at_depot_at_end_penalty=0.0,
            functional_bikes_at_end_penalty=0.0
        )

    @staticmethod
    def benchmark_with_maintenance() -> "RewardConfig":
        """Starvation + congestion + fleet degradation penalty (active maintenance signal)."""
        return RewardConfig(
            weight_starvation=-1.0,
            weight_congestion=-1.0,
            weight_trip_served=0.0,
            weight_fleet_degradation=-0.0,
            not_at_depot_at_end_penalty=0.0,
            functional_bikes_at_end_penalty=0.0,
        )

class RewardCalculator:
    def __init__(self, config: Optional[RewardConfig] = None, gamma: float = 0.99):
        self.config = config or RewardConfig()
        self.gamma = gamma
        
        # --- FIXED: Set to 1.0. Stop crushing the reward signal! ---
        # Have removed scaling due to it crushing the reward signal and causing instability. 
        # The gamma discounting will still be applied during learning updates, so we can 
        # afford to keep the raw reward values intact for better learning dynamics.
        # Plus, the training loop now divides by elapsed_time to create a rate.
        self._scale_factor = 1.0
       
        # Move the state tracking out of the policy and into the calculator
        self._prev_starvations = 0
        self._prev_congestions = 0
        self._prev_trips = 0

    def reset_episode(self):
        """Must be called at the start of every 14-day episode."""
        self._prev_starvations = 0
        self._prev_congestions = 0
        self._prev_trips = 0

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
        """Continuous penalty in final 2 hours of shift when vehicle is away from depot with cargo."""
        if (self.config.not_at_depot_at_end_penalty == 0.0 and
                self.config.functional_bikes_at_end_penalty == 0.0):
            return 0.0
        if vehicle.is_at_depot():
            return 0.0
        from settings import SERVICE_TIME_TO
        close_min = SERVICE_TIME_TO * 60.0
        clock_min = state.time % 1440.0
        time_remaining = max(0.0, close_min - clock_min)
        penalty_window = 120.0
        if time_remaining >= penalty_window:
            return 0.0
        urgency = 1.0 - (time_remaining / penalty_window)
        penalty = self.config.not_at_depot_at_end_penalty * urgency
        if self.config.functional_bikes_at_end_penalty != 0.0:
            n_func = sum(
                1 for b in vehicle.get_bike_inventory()
                if getattr(b, "damage_status", None) not in ("depot", "onsite")
            )
            penalty += self.config.functional_bikes_at_end_penalty * n_func * urgency
        return penalty * self._scale_factor

    def compute_step_reward(self, simulator_metrics, executed_action=None) -> float:
        """Calculates the reward since the last decision epoch."""
        cur_s = simulator_metrics.get_aggregate_value("starvations") or 0
        cur_c = simulator_metrics.get_aggregate_value("long congestions") or 0
        cur_t = simulator_metrics.get_aggregate_value("trips") or 0

        delta_s = cur_s - self._prev_starvations
        delta_c = cur_c - self._prev_congestions
        delta_t = cur_t - self._prev_trips

        reward = 0.0
        reward += self.config.weight_starvation * delta_s
        reward += self.config.weight_congestion * delta_c
        
        if self.config.weight_trip_served != 0.0:
            served = max(0, delta_t - delta_s - delta_c)
            reward += self.config.weight_trip_served * served

        if executed_action is not None and self.config.use_maintenance_shaping:
            if executed_action.bikes_unloaded_for_repair > 0:
                reward += (executed_action.bikes_unloaded_for_repair * 10.0)
            if executed_action.bikes_loaded_from_queue > 0:
                reward += (executed_action.bikes_loaded_from_queue * 5.0)
            if executed_action.bikes_repaired_onsite > 0:
                reward += (executed_action.bikes_repaired_onsite * 5.0)

        # Update trackers for the next step
        self._prev_starvations = cur_s
        self._prev_congestions = cur_c
        self._prev_trips = cur_t

        return reward * self._scale_factor
