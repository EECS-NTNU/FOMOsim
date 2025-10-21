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
            st.leave_intensities = st.leave_intensities.copy() if st.leave_intensities else None
            st.arrive_intensities = st.arrive_intensities.copy() if st.arrive_intensities else None
        
        for a in state.get_areas():
            a.leave_intensities = a.leave_intensities.copy() if a.leave_intensities else None
            a.arrive_intensities = a.arrive_intensities.copy() if a.arrive_intensities else None

    def __repr__(self):
        return f"{self.__class__.__name__}"

