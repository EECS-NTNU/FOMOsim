from progress.bar import Bar

import sim
import os
import sys
import numpy as np
import copy
import settings
import math
import policies

def poisson(k, l):
    return (pow(l, k) * pow(math.e, -l)) / math.factorial(k)

def poisson_le(k, l):
    val = 0
    for i in range(k+1):
        val += poisson(i, l)
    return val

def poisson_gt(k, l):
    return 1 - poisson_le(k, l)

def p_starvation(arrive, leave, target, cap):
    val = poisson_le(target, leave)
    for i in range(1, cap - target):
        val += poisson(target + i, leave) * poisson_gt(i - 1, arrive)
    return 1 - val

def p_congestion(arrive, leave, target, cap):
    val = poisson_le(cap - target, arrive)
    for i in range(1, target):
        val += poisson(cap - target + i, arrive) * poisson_gt(i - 1, leave)
    return 1 - val

def us_target_state(state):
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

    for day in range(7):
        for hour in range(24):
            for st in state.stations:
                cap = st.capacity
                leave = st.get_leave_intensity(day, hour)
                arrive = st.get_arrive_intensity(day, hour)

                min_diff = 1
                min_target = 0
                prev_diff = 1
                for target in range(cap+1):
                    diff = abs(p_starvation(arrive, leave, target, cap) - p_congestion(arrive, leave, target, cap))
                    if diff < min_diff:
                        min_diff = diff
                        min_target = target
                    if prev_diff < diff:
                        # this should be safe, because p_starvation is always decreasing, and p_congestion is always increasing
                        break 
                    prev_diff = diff

                target_state[st.id][day][hour] = min_target

            progress.next()

    progress.finish()

    return target_state
