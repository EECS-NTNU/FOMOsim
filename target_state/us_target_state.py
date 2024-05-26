import sim
from target_state import TargetState
import numpy as np

class USTargetState(TargetState):

    def __init__(self):
        super().__init__()

    def update_target_state(self, state, day, hour):
        num_sb_bikes = len(state.get_all_sb_bikes())
        num_stations = len(state.get_stations())

        for st in state.get_stations():
            cap = st.capacity
            leave = st.leave_intensities[day][hour]
            arrive = st.arrive_intensities[day][hour]
            leave_std = st.leave_intensities_stdev[day][hour]
            arrive_std = st.arrive_intensities_stdev[day][hour]

            if (leave_std==0) or (arrive_std==0):
                ts = num_sb_bikes // num_stations
            else:
                # ts = (leave + leave_std - (arrive - arrive_std)) + (num_sb_bikes // num_stations)
                ts = (leave_std*(cap-arrive)+arrive_std*leave)/(leave_std+arrive_std)
            st.target_state[day][hour] = ts
        
        for area in state.get_areas():
            leave = area.leave_intensities[day][hour]
            arrive = area.arrive_intensities[day][hour]
            leave_std = area.leave_intensities_stdev[day][hour]
            arrive_std = area.arrive_intensities_stdev[day][hour]
            area.target_state[day][hour] = max(0, round((leave + leave_std - (arrive - arrive_std))) + state.rng.randint(0, 2))
