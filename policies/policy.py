"""
This file contains the base policy class
"""
import copy

import sim
import numpy.random as random
import abc

class Policy(abc.ABC):
    """
    Base Policy class. Used mainly as an interface for the get best action method
    """

    @abc.abstractmethod
    def get_best_action(self, world, vehicle):
        """
        Returns the best action for the input vehicle in the world context
        :param world: world object that contains the whole world state
        :param vehicle: the vehicle to perform an action
        :return: the best action according to the policy
        """
        pass

    def setup_from_state(self, state):
        """
        Function to be called after association with a state object is created.
        Nice place to setup value functions.
        :param state: state object associated with policy
        """
        pass

    def initSim(self, sim):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

