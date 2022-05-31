import sim
from progress.bar import Bar
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

def poisson_ge(k, l):
    return 1 - poisson_le(k - 1, l)

def poisson_gt(k, l):
    return 1 - poisson_le(k, l)

def us_ideal_state(state):
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
            total_ideal_state = 0

            for st in state.stations:
                cap = st.capacity
                leave = st.get_leave_intensity(day, hour)
                arrive = st.get_arrive_intensity(day, hour)

                diff = 1
                for ideal in range(cap):
                    next_diff = abs(poisson_gt(ideal, leave) - poisson_ge(cap - ideal, arrive))
                    if next_diff > diff:
                        ideal = ideal - 1
                        break
                    diff = next_diff

                print(ideal)

                ideal_state[st.id][day][hour] = ideal

    return ideal_state
