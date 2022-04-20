import clustering.scripts
import sim
from progress.bar import Bar
import os
import numpy as np
import copy
import settings
import math
import policies

MIN_IDEAL_STATE = 5

def outflow_ideal_state(state):
    # initialize ideal_state matrix
    ideal_state = []
    for st in state.locations:
        ideal_state.append([])
        for day in range(7):
            ideal_state[st.id].append([])
            for hour in range(24):
                ideal_state[st.id][day].append(0)

    for day in range(7):
        for hour in range(24):
            # set ideal state to net outflow of bikes
            total_ideal_state = 0
            num_zero = 0
            for st in state.stations:
                outflow = st.get_leave_intensity(day, hour) - st.get_arrive_intensity(day, hour)
                if outflow >= 0:
                    ideal_state[st.id][day][hour] = outflow
                    total_ideal_state += outflow
                else:
                    ideal_state[st.id][day][hour] = 0
                    num_zero += 1

            # scale ideal states so that sum is close to total number of scooters
            if total_ideal_state > 0:
                scale_factor = (len(state.get_all_scooters()) - (num_zero * MIN_IDEAL_STATE)) / total_ideal_state
                for st in state.stations:
                    ideal_state[st.id][day][hour] = int(ideal_state[st.id][day][hour] * scale_factor)

            # make sure all stations have at least MIN_IDEAL_STATE
            for st in state.stations:
                if ideal_state[st.id][day][hour] < MIN_IDEAL_STATE:
                    ideal_state[st.id][day][hour] = MIN_IDEAL_STATE

    return ideal_state
