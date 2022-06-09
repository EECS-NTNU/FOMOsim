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

def fosen_haldorsen_target_state(state):
    # initialize target_state matrix
    target_state = []
    for st in state.locations:
        target_state.append([])
        for day in range(7):
            target_state[st.id].append([])

    #target states are defined externally
    with open('init_state//fosen_haldorsen//station.json', 'r') as f:
        target_state_json = json.load(f)

    for st in state.locations:
        target = {}
        if st.original_id in target_state_json.keys():
            target = {int(k): int(v) for k, v in target_state_json[st.original_id].items()}
        else:
            for hour in range(0, 24):
                target[hour] = st.capacity // 2

        for hour in range(24):
            target_state[st.id][0].append(target[hour])

    for day in range(1, 7):
        target_state[st.id][day] = target_state[st.id][0]

    return target_state
