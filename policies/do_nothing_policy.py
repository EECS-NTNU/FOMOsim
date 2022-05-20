"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import abc

class DoNothing(Policy):
    def __init__(self):
        super().__init__()

    def get_best_action(self, simul, vehicle):
        return sim.Action([], [], [], 0)

    def init_sim(self, simul):
        # Empty the vehicle arrival events in the event_queue
        simul.event_queue = [
            event
            for event in simul.event_queue
            if not isinstance(event, sim.VehicleArrival)
        ]
        
