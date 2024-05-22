"""
This file contains a do nothing policy
"""

from policies import Policy
import sim
from settings import *

class DoNothing(Policy):
    def __init__(self,
                 max_depth = MAX_DEPTH, 
                number_of_successors = NUM_SUCCESSORS, 
                time_horizon = TIME_HORIZON, 
                criticality_weights_set = CRITICAILITY_WEIGHTS_SET_SB, 
                evaluation_weights = EVALUATION_WEIGHTS, 
                number_of_scenarios = NUM_SCENARIOS, 
                discounting_factor = DISCOUNTING_FACTOR,
                overflow_criteria = OVERFLOW_CRITERIA,
                starvation_criteria = STARVATION_CRITERIA,
                swap_threshold = BATTERY_LIMIT_TO_SWAP
                 ):
        self.max_depth = max_depth
        self.number_of_successors = number_of_successors
        self.time_horizon = time_horizon
        self.criticality_weights_set = criticality_weights_set
        self.evaluation_weights = evaluation_weights
        self.number_of_scenarios = number_of_scenarios
        self.discounting_factor = discounting_factor
        self.overflow_criteria = overflow_criteria
        self.starvation_criteria = starvation_criteria
        self.swap_threshold = swap_threshold
        super().__init__()

    def get_best_action(self, simul, vehicle):
        bikes_to_swap = []
        bikes_to_pickup = []
        bikes_to_deliver = []

        next_location_id = simul.state.rng.choice([loc.location_id for loc in simul.state.get_locations() if loc != vehicle.location])

        action = sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_location_id,
        )

        return action