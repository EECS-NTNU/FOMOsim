import sim
from target_state import TargetState

class HalfCapacityTargetState(TargetState):

    def __init__(self):
        super().__init__()

    def update_target_state(self, state, day, hour):
        num_bikes = len(state.get_all_bikes())
        num_stations = len(state.locations)

        for st in state.locations:
            if isinstance(st, sim.Depot):
                st.target_state = 0
            else:
                st.target_state = st.capacity/2
