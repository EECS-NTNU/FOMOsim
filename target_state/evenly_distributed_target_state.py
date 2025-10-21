import sim
from target_state import TargetState

class EvenlyDistributedTargetState(TargetState):

    def __init__(self):
        super().__init__()

    def update_target_state(self, state, day, hour):
        num_bikes = len(state.get_all_bikes())
        num_stations = len(state.stations)

        for st in state.get_stations():
            if isinstance(st, sim.Depot):
                st.target_state = 0
            else:
                st.target_state = num_bikes // num_stations
