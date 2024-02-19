import sim
from target_state import TargetState

class USTargetState(TargetState):

    def __init__(self):
        super().__init__()

    def update_target_state(self, state, day, hour):
        num_bikes = len(state.get_all_bikes())
        num_stations = len(state.locations)

        for st in state.get_locations():
            cap = st.capacity
            leave = st.leave_intensities[day][hour]
            arrive = st.arrive_intensities[day][hour]
            leave_std = st.leave_intensities_stdev[day][hour]
            arrive_std = st.arrive_intensities_stdev[day][hour]

            if (leave_std==0) or (arrive_std==0):
                ts = num_bikes // num_stations
            else:
                ts = (leave + leave_std - (arrive - arrive_std)) + (num_bikes // num_stations)
                #ts = (leave_std*(cap-arrive)+arrive_std*leave)/(leave_std+arrive_std)
            st.target_state[day][hour] = ts
