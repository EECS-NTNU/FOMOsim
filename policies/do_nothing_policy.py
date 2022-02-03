"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import numpy.random as random
import abc

class DoNothing(Policy):
    def __init__(self):
        super().__init__(0, 0)

    def get_best_action(self, world, vehicle):
        return sim.Action([], [], [], 0)

    def initSim(self, simul):
        # Empty the vehicle arrival events in the stack
        simul.stack = [
            event
            for event in simul.stack
            if not isinstance(event, sim.VehicleArrival)
        ]
        
