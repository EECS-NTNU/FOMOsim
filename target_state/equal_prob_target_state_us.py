from progress.bar import Bar

import sim
import os
import sys
import numpy as np
import copy
import settings
import math
import policies


def equal_prob_target_state_us(state):
    # initialize target_state matrix
    target_state = []
    for st in state.locations:
        target_state.append([])
        for day in range(7):
            target_state[st.id].append([])
            for hour in range(24):
                target_state[st.id][day].append(0)

    progress = Bar(
        "Calculating target state",
        max = 7 * 24
    )

    num_bikes = len(state.get_all_bikes())
    num_stations = len(state.stations)

    for day in range(7):
        for hour in range(24):
            for st in state.stations.values():
                cap = st.capacity
                leave = st.get_leave_intensity(day, hour)
                arrive = st.get_arrive_intensity(day, hour)
                leave_std = st.leave_intensities_std[day % 7][hour % 24]
                arrive_std = st.arrive_intensities_std[day % 7][hour % 24]

                if (leave_std==0) or (arrive_std==0):
                    ts = num_bikes // num_stations  #why not half the capacity
                else:
                    ts = round((leave_std*(cap-arrive)+arrive_std*leave)/(leave_std+arrive_std)) # do we need to round? No!
                target_state[st.id][day][hour] = ts

            progress.next()

    progress.finish()

    return target_state
