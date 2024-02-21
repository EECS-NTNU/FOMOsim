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
        # default implementation just copies historical demand
        for st in state.get_stations():
            st.leave_intensities = st.leave_intensities.copy()
            st.arrive_intensities = st.arrive_intensities.copy()
        
        for a in state.get_areas():
            a.leave_intensities = a.leave_intensities.copy()
            a.arrive_intensities = a.arrive_intensities.copy()

    def __repr__(self):
        return f"{self.__class__.__name__}"

