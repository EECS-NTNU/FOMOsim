"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import abc

class NightShift(Policy):
    def __init__(self):
        super().__init__()

    def get_best_action(self, simul, vehicle):
        return sim.Action([], [], [], 0)

    def initSim(self, simul):
        # TODO: Do every night
        simul.event_queue = [
            event
            for event in simul.event_queue
            if not isinstance(event, sim.VehicleArrival)
        ]
        for cluster in simul.state.stations:
            for scooter in cluster.scooters:
                if isinstance(scooter, sim.Scooter):
                    if scooter.battery < 70:
                        scooter.swap_battery()
