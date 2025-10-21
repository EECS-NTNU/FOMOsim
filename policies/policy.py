"""
This file contains the base policy class
"""

import abc

from .action import Action
import settings

class Policy(abc.ABC):
    """
    Base Policy class
    """

    def __init__(self):
        self.set_time_of_service()

    @abc.abstractmethod
    def get_best_action(self, state, vehicle):
        """
        Returns the best action for the input vehicle in the simul context
        :param state: object that contains the whole simulation state
        :param vehicle: the vehicle to perform an action
        :return: the best action according to the policy (instance of the Action class in sim/Action.py)
        """
        pass

    def set_time_of_service(self, hour_from=settings.SERVICE_TIME_FROM, hour_to=settings.SERVICE_TIME_TO):
        self.hour_from = hour_from
        self.hour_to = hour_to

    def get_action(self, state, vehicle):
        """
        This is called from the event simulator.  Policies should implement get_best_action(), not get_action()
        """

        if  self.hour_from <= state.hour() < self.hour_to: 
            return self.get_best_action(state, vehicle)
        else:
            depot_id = simul.state.get_closest_depot(vehicle)
            return sim.Action([], [], [], depot_id)

    def __repr__(self):
        return f"{self.__class__.__name__}"

