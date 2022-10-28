import sim
from target_state import TargetState

class USTargetState(TargetState):

    def __init__(self):
        super().__init__()

    def update_target_state(self, state, day, hour):
        num_bikes = len(state.get_all_bikes())
        num_stations = len(state.stations)

        for st in state.stations.values():
            cap = st.capacity
            leave = st.historical_leave_intensities[day][hour]
            arrive = st.historical_arrive_intensities[day][hour]
            leave_std = st.historical_leave_intensities_stdev[day][hour]
            arrive_std = st.historical_arrive_intensities_stdev[day][hour]

            if (leave_std==0) or (arrive_std==0):
                ts = st.capacity/2 #num_bikes // num_stations
            else:
                ts = (leave_std*(cap-arrive)+arrive_std*leave)/(leave_std+arrive_std)
            st.target_state[day][hour] = ts