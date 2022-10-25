"""
This file contains the base target state class
"""
import copy

import sim
import abc
from settings import *

class TargetState(abc.ABC):
    """
    Base Target State class
    """

    def __init__(self):
        pass

    def update_target_state(self, state, day, hour):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

