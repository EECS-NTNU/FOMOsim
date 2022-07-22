"""
This file contains a random action policy
"""
import copy
import math

from policies import Policy, neighbour_filtering
import sim
import abc
from sim import State, Vehicle

class RandomActionPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_best_action(self, simul, vehicle):
        bikes_to_swap = []
        bikes_to_pickup = []
        bikes_to_deliver = []

        next_location_id = simul.state.rng.choice([i for i in range(len(simul.state.locations)) if i != vehicle.location.id])

        if not vehicle.is_at_depot():
            num_deliver = 0
            max_deliver = min(len(vehicle.bike_inventory),
                              vehicle.location.capacity - len(vehicle.location.bikes))
            if max_deliver > 0:
                num_deliver = simul.state.rng.integers(0, max_deliver)

            num_pickup = 0
            max_pickup = min(len(vehicle.location.bikes),
                             vehicle.bike_inventory_capacity - len(vehicle.bike_inventory), vehicle.battery_inventory)
            if max_pickup > 0:
                num_pickup = simul.state.rng.integers(0, max_pickup)

            if num_pickup > 0:
                pickable = list(vehicle.location.bikes.keys())
                bikes_to_pickup = simul.state.rng.choice(pickable, num_pickup, replace=False)

            if num_deliver > 0:
                deliverable = [ bike.id for bike in vehicle.get_bike_inventory() ]
                bikes_to_deliver = simul.state.rng.choice(deliverable, num_deliver, replace=False)

            swappable = [ bike.id for bike in vehicle.location.get_swappable_bikes() if bike.id not in bikes_to_pickup ]

            num_swap = 0
            max_swap = max(0, min(len(swappable), vehicle.battery_inventory - num_pickup))
            if max_swap > 0:
                num_swap = simul.state.rng.integers(0, max_swap)

            if num_swap > 0:
                bikes_to_swap = simul.state.rng.choice(swappable, num_swap, replace=False)

        action = sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_location_id,
        )

        return action
