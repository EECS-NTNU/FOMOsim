import sim

def evenly_distributed_ideal_state(state):
    num_bikes = len(state.get_all_scooters())
    num_stations = len(state.locations)
    ideal_state = []
    for st in state.locations:
        ideal_state.append([])
        for day in range(7):
            ideal_state[st.id].append([])
            for hour in range(24):
                ideal_state[st.id][day].append(num_bikes // num_stations)
    return ideal_state
