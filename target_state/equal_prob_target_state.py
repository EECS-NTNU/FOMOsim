import math
from target_state import TargetState
 
 
def _poisson(k, l):
    return (pow(l, k) * pow(math.e, -l)) / math.factorial(k)
 
def _poisson_le(k, l):
    val = 0.0
    for i in range(k + 1):
        val += _poisson(i, l)
    return val
 
def _poisson_gt(k, l):
    return 1.0 - _poisson_le(k, l)
 
def _p_starvation(arrive, leave, target, cap):
    val = _poisson_le(target, leave)
    for i in range(1, cap - target):
        val += _poisson(target + i, leave) * _poisson_gt(i - 1, arrive)
    return 1.0 - val
 
def _p_congestion(arrive, leave, target, cap):
    val = _poisson_le(cap - target, arrive)
    for i in range(1, target):
        val += _poisson(cap - target + i, arrive) * _poisson_gt(i - 1, leave)
    return 1.0 - val
 
def _find_equal_prob_target(arrive, leave, cap):
    """Find the target that minimises |P(starvation) - P(congestion)|."""
    min_diff = 1.0
    min_target = 0
    prev_diff = 1.0
    for target in range(cap + 1):
        diff = abs(_p_starvation(arrive, leave, target, cap)
                   - _p_congestion(arrive, leave, target, cap))
        if diff < min_diff:
            min_diff = diff
            min_target = target
        if prev_diff < diff:
            break
        prev_diff = diff
    return min_target
 
 
class EqualProbTargetState(TargetState):
    # Class level cache {instance_key -> {station_id -> [[target] * 24] * 7}}
    _cache: dict = {}
 
    @classmethod
    def _instance_key(cls, state):
        return frozenset((st.id, st.capacity) for st in state.get_stations())
 
    @classmethod
    def _build_cache(cls, state, key):
        cache = {}
        for st in state.get_stations():
            cache[st.id] = [[0] * 24 for _ in range(7)]
        for day in range(7):
            for hour in range(24):
                for st in state.get_stations():
                    arrive = st.get_arrive_intensity(day, hour)
                    leave = st.get_leave_intensity(day, hour)
                    cache[st.id][day][hour] = _find_equal_prob_target(
                        arrive, leave, st.capacity
                    )
        cls._cache[key] = cache
 
    @classmethod
    def _prime_state(cls, state, cache):
        for st in state.get_stations():
            if st.id in cache:
                for day in range(7):
                    for hour in range(24):
                        st.target_state[day][hour] = cache[st.id][day][hour]
 
    def update_target_state(self, state, day, hour):
        key = self._instance_key(state)
        if key not in self._cache:
            self._build_cache(state, key)
        # Check the object directly instead of using a global id() set
        if not getattr(state, '_targets_primed', False):
            self._prime_state(state, self._cache[key])
            state._targets_primed = True