"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import numpy.random as random
import abc

import system_simulation.scripts

class NightShift(Policy):
    def __init__(self):
        super().__init__(0, 0)

    def get_best_action(self, world, vehicle):
        return sim.Action([], [], [], 0)

    def initWorld(self, world):
        world.stack = [
            event
            for event in world.stack
            if not isinstance(event, sim.VehicleArrival)
        ]
        for cluster in world.state.clusters:
            for scooter in cluster.scooters:
                if scooter.battery < 70:
                    scooter.swap_battery()
