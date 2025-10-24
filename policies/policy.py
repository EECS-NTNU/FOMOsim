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

    def __init__(self):
        self.set_time_of_service()

    @abc.abstractmethod
    #def get_best_action(self, simul, vehicle):
    def get_best_action(self, state, vehicle):
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

    #def get_action(self, simul, vehicle):
    def get_action(self, simul, vehicle):
        """
        This is called from the event simulator.  Policies should implement get_best_action(), not get_action()
        """

        if  self.hour_from <= simul.hour() < self.hour_to: 
            return self.get_best_action(simul, vehicle)
        else:
            depot_id = simul.state.get_closest_depot(vehicle)
            return sim.Action([], [], [], depot_id)

    def __repr__(self):
        return f"{self.__class__.__name__}"

