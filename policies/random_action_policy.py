"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import classes
import numpy.random as random
import abc

import system_simulation.scripts

class RandomActionPolicy(Policy):
    def __init__(self, get_possible_actions_divide, number_of_neighbors):
        super().__init__(get_possible_actions_divide, number_of_neighbors)

    def get_best_action(self, world, vehicle):
        # all possible actions in this state
        possible_actions = world.state.get_possible_actions(
            vehicle,
            exclude=world.tabu_list,
            time=world.time,
            divide=self.get_possible_actions_divide,
            number_of_neighbours=self.number_of_neighbors,
        )

        # pick a random action
        if len(possible_actions) > 0:
          return random.choice(possible_actions)
        else:
          return classes.Action([], [], [], 0)
