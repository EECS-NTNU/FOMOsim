import sim
from target_state import TargetState

import math

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

class EqualProbTargetState(TargetState):

    def __init__(self):
        super().__init__()

    def update_target_state(self, state, day, hour):
        for st in state.stations.values():
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
                    break 
                prev_diff = diff

            st.target_state[day][hour] = min_target
