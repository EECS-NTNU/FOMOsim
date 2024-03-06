import sim
from target_state import TargetState

class OutflowTargetState(TargetState):

    def __init__(self):
        super().__init__()

    def update_target_state(self, state, day, hour):
        # set ideal state to net outflow of bikes
        total_target_state = 0

        for st in state.locations:
            outflow = st.get_leave_intensity(day, hour) - st.get_arrive_intensity(day, hour)
            if outflow > 0:
                st.target_state = 2
                total_target_state += 2
            else:
                st.target_state = 1
                total_target_state += 1

        # scale ideal states so that sum is close to total number of bikes
        scale_factor = len(state.get_all_bikes()) / total_target_state
        for st in state.locations:
            target = int(st.target_state * scale_factor)
            if target > (0.7 * st.capacity):
                target = 0.7 * st.capacity
            if target < (0.3 * st.capacity):
                target = 0.3 * st.capacity
            st.target_state = target
