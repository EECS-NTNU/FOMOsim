import json
from init_state.fosen_haldorsen.set_up_simulation_data import setup_stations_students

def get_initial_state(init_hour=8, number_of_vans=1, random_seed=1):
    state = setup_stations_students('uip-students', init_hour, number_of_vans, random_seed, False)

    for st in state.stations:
        if int(st.original_id) % 10 == 0:
            st.battery_rate = 1
        else:
            st.battery_rate = 0.95

    create_subset(state.stations, len(state.stations))
    print("UIP DB objects collected")
    
    return state

def create_subset(stations_uip,n):

    demand_met = 1

    subset = stations_uip[:n]
    subset_ids = [s.id for s in subset]

    for st1 in subset:
        for day in range(7):
            for hour in range(24):
                subset_prob = 0
                for st_id, prob in enumerate(st1.move_probabilities[day][hour]):
                    if st_id in subset_ids:
                        subset_prob += prob
                st1.leave_intensity_per_iteration[day][hour] *= subset_prob
            for s_id in subset_ids:
                st1.move_probabilities[day][hour][s_id] /= subset_prob

    for day in range(7):
        for hour in range(24):
            for st2 in subset:
                incoming = 0
                for stat in subset:
                    incoming += stat.leave_intensity_per_iteration[day][hour] * demand_met * stat.move_probabilities[day][hour][st2.id]
                st2.arrive_intensity_per_iteration[day][hour] = incoming
