"""
This file contains the base policy class
"""
import copy

import sim
import abc

class Policy(abc.ABC):
    """
    Base Policy class. Used mainly as an interface for the get best action method
    """

    @abc.abstractmethod
    def get_best_action(self, simul, vehicle):
        """
        Returns the best action for the input vehicle in the world context
        :param simul: simulator object that contains the whole simulation state
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
        """
        Function called after sim object is created
        For initializing stuff that needs access to the initial event queue or state
        :param sim: simulator object
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

