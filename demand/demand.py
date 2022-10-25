"""
This file contains the base demand class
"""
import copy

import sim
from settings import *

class Demand():
    """
    Base Demand class
    """

    def __init__(self):
        pass

    def update_demands(self, state, day, hour):
        for st in state.locations:
            st.leave_intensities = st.historical_leave_intensities.copy()
            st.arrive_intensities = st.historical_arrive_intensities.copy()

    def __repr__(self):
        return f"{self.__class__.__name__}"

