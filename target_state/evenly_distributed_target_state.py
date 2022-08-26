import sim

def evenly_distributed_target_state(state):
    num_bikes = len(state.get_all_bikes())
    num_stations = len(state.stations)
    target_state = []
    for st in state.locations:
        target_state.append([])
        for day in range(7):
            target_state[st.id].append([])
            for hour in range(24):
                if isinstance(st, sim.Depot):
                    target_state[st.id][day].append(0)
                else:
                    target_state[st.id][day].append(num_bikes // num_stations)
    return target_state

