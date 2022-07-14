import json
from init_state.fosen_haldorsen.set_up_simulation_data import setup_stations_students

def get_initial_state(init_hour=8, number_of_vehicles=1, random_seed=1, number_of_stations=None):
    state = setup_stations_students('uip-students', init_hour, number_of_vehicles, random_seed, False)

    if number_of_stations is None:
        number_of_stations = len(state.locations)

    create_subset(state, number_of_stations)
    print("UIP DB objects collected")
    
    return state

def create_subset(state, n):

    demand_met = 1

    subset = state.locations[:n]
    subset_ids = [s.id for s in subset]

    for station in subset:
        for day in range(7):
            for hour in range(24):
                station.move_probabilities[day][hour] = station.move_probabilities[day][hour][:n]

                subset_prob = 0
                for prob in station.move_probabilities[day][hour]:
                    subset_prob += prob

                station.leave_intensity_per_iteration[day][hour] *= subset_prob

                for destination_id in subset_ids:
                    station.move_probabilities[day][hour][destination_id] /= subset_prob

    for day in range(7):
        for hour in range(24):
            for station in subset:
                incoming = 0
                for from_station in subset:
                    incoming += from_station.leave_intensity_per_iteration[day][hour] * demand_met * from_station.move_probabilities[day][hour][station.id]
                station.arrive_intensity_per_iteration[day][hour] = incoming

    state.set_locations(subset)
