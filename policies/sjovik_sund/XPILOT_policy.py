import sys
from pathlib import Path
import os
import sim
from policies import Policy

# Import the existing X- PILOT based on the PILOT implementation from inngjerdingen_moeller, which is designed with neighborhood interaction for normal bikes.
from policies.inngjerdingen_moeller.PILOT_policy import PILOT

class XPILOTPolicy(Policy):
    """
    A wrapper to run the established X-PILOT (or base PILOT) method 
    *WITHOUT* neighborhood interaction. 
    
    This wraps the BS_PILOT (Station-Based PILOT) from inngjerdingen_moeller, which is designed for e-bike fleet management with swappable batteries.
    making it plug-and-play for benchmarking against the VFA/MDP implementations 
    in sjovik_sund.
    """
    
    def __init__(self, time_horizon=40, max_depth=2, num_successors=5, number_of_scenarios=100, maintenance_enabled=False):
        """
        Args:
            time_horizon: Forward simulation lookahead duration (minutes).
            max_depth: Maximum tree depth for the forward search.
            num_successors: Number of successor states to consider at each node.
            num_scenarios: Number of demand scenarios to evaluate.
            maintenance_enabled: Whether to adapt actions for component failures (if supported).
        """
        super().__init__()
        self.time_horizon = time_horizon
        self.maintenance_enabled = maintenance_enabled
        
        # Instantiate the pure PILOT without collaborative neighbour filtering
        self.pilot_backend = PILOT(
            time_horizon=time_horizon,
            max_depth=max_depth,
            number_of_successors=num_successors,
            number_of_scenarios= number_of_scenarios,
            enable_neighborhood_interactions=False,  # Disable neighborhood interactions for pure PILOT behavior
        )

    def get_best_action(self, state: sim.State, vehicle: sim.Vehicle) -> sim.Action:
        """
        Delegate the decision-making process backward to the base PILOT implementation.
        """
        # Get action from the backend PILOT
        base_action = self.pilot_backend.get_best_action(state, vehicle)
        
        # If component maintenance (not just battery swaps) is dynamically tracked,
        # we can optionally map the expected action dictionary here if needed,
        # but pure PILOT handles standard bike swaps + routing.
        return base_action

    @property
    def weights(self):
        """Mock weights property for logging compatibility in run_simulation.py."""
        return [0, 0, 0, 0]
