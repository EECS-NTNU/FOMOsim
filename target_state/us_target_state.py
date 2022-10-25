import sim
from progress.bar import Bar
import os
import sys
import numpy as np
import copy
import settings
import math
import policies
import json

def us_target_state(state):
    # initialize target_state matrix
    target_state = []
    for st in state.locations:
        target_state.append([])
        for day in range(7):
            target_state[st.id].append([])

    for st in state.locations:
        target = {}

        for hour in range(0, 24):
            target[hour] = st.capacity // 2

        for hour in range(24):
            target_state[st.id][0].append(target[hour])

        for day in range(1, 7):
            target_state[st.id][day] = target_state[st.id][0]

    return target_state
