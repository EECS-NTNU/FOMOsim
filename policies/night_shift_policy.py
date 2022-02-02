"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import numpy.random as random
import abc

class NightShift(Policy):
    def __init__(self):
        super().__init__(0, 0)

    def get_best_action(self, world, vehicle):
        return sim.Action([], [], [], 0)

    def initSim(self, sim):
        sim.stack = [
            event
            for event in sim.stack
            if not isinstance(event, sim.VehicleArrival)
        ]
        for cluster in sim.state.stations:
            for scooter in cluster.scooters:
                if scooter.battery < 70:
                    scooter.swap_battery()
