from target_state import TargetState
import numpy as np
import json
import gzip

class HLVTargetState(TargetState):

    def __init__(self,
                 target_state_ff_json):
        super().__init__()
        self.target_state_ff = self.load_target_ff(target_state_ff_json)

    def load_target_ff(self, target_state_file):
        with gzip.open(f'{target_state_file}', 'r') as file:
            data = json.load(file)
        target_state_dict = {area['id']: area['target_states'] for area in data['areas']}
        return target_state_dict
    
    def set_target_states(self, state):
        for day in range(7):
            for hour in range(24):
                self.set_sb_targets(state, day, hour)
        
        for area in state.get_areas():
            area.target_state = self.target_state_ff[area.location_id]
    
    def set_sb_targets(self, state, day, hour):
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
            area.target_state[day][hour] = self.target_state_ff[area.location_id][day][hour]
