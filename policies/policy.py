"""
This file contains the base policy class
"""
import copy

import sim
import abc
from settings import *

class Policy(abc.ABC):
    """
    Base Policy class
    """

    @abc.abstractmethod
    def get_best_action(self, simul, vehicle):
        """
        Returns the best action for the input vehicle in the world context
        :param simul: simulator object that contains the whole simulation state
        :param vehicle: the vehicle to perform an action
        :return: the best action according to the policy (instance of the Action class in sim/Action.py)
        """
        pass

    def init_sim(self, sim):
        """
        Function called after sim object is created
        For initializing stuff that needs access to the initial event queue or state
        :param sim: simulator object
        """
        pass

    def set_time_of_service(self,hour_from=SERVICE_TIME_FROM, hour_to=SERVICE_TIME_TO):
        self.hour_from = hour_from
        self.hour_to = hour_to

    def __repr__(self):
        return f"{self.__class__.__name__}"

