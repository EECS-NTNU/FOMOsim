import json
from init_state.fosen_haldorsen.set_up_simulation_data import setup_stations_students

def get_initial_state(init_hour=8, number_of_vehicles=1, random_seed=1):
    state = setup_stations_students('uip-students', init_hour, number_of_vehicles, random_seed, False)
    print("UIP DB objects collected")
    return state
