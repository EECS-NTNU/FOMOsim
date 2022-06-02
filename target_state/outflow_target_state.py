import sim
from progress.bar import Bar
import os
import sys
import numpy as np
import copy
import settings
import math
import policies

def outflow_target_state(state):
    # initialize target_state matrix
    target_state = []
    for st in state.locations:
        target_state.append([])
        for day in range(7):
            target_state[st.id].append([])
            for hour in range(24):
                target_state[st.id][day].append(0)

    for day in range(7):
        for hour in range(24):
            # set ideal state to net outflow of bikes
            total_target_state = 0

            for st in state.stations:
                outflow = st.get_leave_intensity(day, hour) - st.get_arrive_intensity(day, hour)
                if outflow > 0:
                    target_state[st.id][day][hour] = 2
                    total_target_state += 2
                else:
                    target_state[st.id][day][hour] = 1
                    total_target_state += 1

            # scale ideal states so that sum is close to total number of scooters
            scale_factor = len(state.get_all_scooters()) / total_target_state
            for st in state.stations:
                target = int(target_state[st.id][day][hour] * scale_factor)
                if target > (0.7 * st.capacity):
                    target = 0.7 * st.capacity
                if target < (0.3 * st.capacity):
                    target = 0.3 * st.capacity
                target_state[st.id][day][hour] = target

    return target_state
