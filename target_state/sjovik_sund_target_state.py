from target_state import TargetState


class SjovikSundTargetState(TargetState):
    """
    Target state for SB (station-based) systems.
    Sets each station's target using a demand-weighted formula:
        ts = (leave_std*(cap - arrive) + arrive_std*leave) / (leave_std + arrive_std)
    Falls back to even distribution when std devs are zero.
    No area handling — SB system has stations only.
    """

    def __init__(self):
        super().__init__()

    def set_target_states(self, state):
        for day in range(7):
            for hour in range(24):
                self._set_station_targets(state, day, hour)

    def _set_station_targets(self, state, day, hour):
        num_sb_bikes = len(state.get_all_sb_bikes())
        num_stations = len(state.get_stations())

        for st in state.get_stations():
            cap        = st.capacity
            leave      = st.leave_intensities[day][hour]
            arrive     = st.arrive_intensities[day][hour]
            leave_std  = st.leave_intensities_stdev[day][hour]
            arrive_std = st.arrive_intensities_stdev[day][hour]

            if leave_std == 0 or arrive_std == 0:
                ts = num_sb_bikes // num_stations
            else:
                ts = (leave_std * (cap - arrive) + arrive_std * leave) / (leave_std + arrive_std)

            st.target_state[day][hour] = ts

    def update_target_state(self, state, day, hour):
        self._set_station_targets(state, day, hour)
