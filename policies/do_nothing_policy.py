"""
This file contains a do nothing policy
"""

from policies import Policy
import sim

class DoNothing(Policy):
    def __init__(self):
        super().__init__()

    def get_best_action(self, simul, vehicle):
        bikes_to_swap = []
        bikes_to_pickup = []
        bikes_to_deliver = []

        next_location_id = simul.state.rng.choice([i for i in range(len(simul.state.locations)) if i != vehicle.location.location_id])

        action = sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_location_id,
        )

        return action