"""
This file contains the base policy class
"""
import copy

import sim
import numpy.random as random
import abc

import system_simulation.scripts


class Policy(abc.ABC):
    """
    Base Policy class. Used mainly as an interface for the get best action method
    """

    def __init__(
        self,
        get_possible_actions_divide,
        number_of_neighbors,
    ):
        self.get_possible_actions_divide = get_possible_actions_divide
        self.number_of_neighbors = number_of_neighbors

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

    def initWorld(self, world):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @staticmethod
    def print_action_stats(
        world,
        vehicle: sim.Vehicle,
        actions_info: [(sim.Action, int, int)],
    ) -> None:
        if world.verbose:
            print(f"\n{vehicle}:")
            for action, reward, computational_time in actions_info:
                print(
                    f"\n{action} Reward - {round(reward, 3)} | Comp. time - {round(computational_time, 2)}"
                )
            print("\n----------------------------------------------------------------")

