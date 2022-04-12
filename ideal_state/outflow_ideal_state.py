import clustering.scripts
import sim
from progress.bar import Bar
import os
import numpy as np
import copy
import settings
import math
import policies

MIN_IDEAL_STATE = 1

def outflow_ideal_state(state):
    # initialize ideal_state matrix
    for cluster in state.stations:
        cluster.ideal_state = []
        for day in range(7):
            cluster.ideal_state.append([])
            for hour in range(24):
                cluster.ideal_state[day].append(0)

    for day in range(7):
        for hour in range(24):
            # set ideal state to net outflow of bikes
            total_ideal_state = 0
            num_zero = 0
            for st in state.stations:
                outflow = st.get_leave_intensity(day, hour) - st.get_arrive_intensity(day, hour)
                if outflow >= 0:
                    st.ideal_state[day][hour] = outflow
                    total_ideal_state += outflow
                else:
                    st.ideal_state[day][hour] = 0
                    num_zero += 1

            # scale ideal states so that sum is close to total number of scooters
            scale_factor = (len(state.get_all_scooters()) - (num_zero * MIN_IDEAL_STATE)) / total_ideal_state
            for st in state.stations:
                st.ideal_state[day][hour] = int(st.ideal_state[day][hour] * scale_factor)

            # make sure all stations have at least MIN_IDEAL_STATE
            for st in state.stations:
                if st.ideal_state[day][hour] < MIN_IDEAL_STATE:
                    st.ideal_state[day][hour] = MIN_IDEAL_STATE
