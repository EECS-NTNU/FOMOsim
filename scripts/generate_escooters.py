import json
from hexweb import HexWeb
import gzip
import random
import multiprocessing as mp

def generate_escooters():
    # generate 
    WEEKS = 3
    resolutions = [11, 10, 9, 8]
    ring_radiuss = [100, 58, 22, 10]

    for i in range(4):
        resolution = resolutions[i]
        ring_radius = ring_radiuss[i]

        FINISHED_DATA_FILE = f'/Users/itlam/fomo/instances/Ryde/TD_res{resolution}_radius{ring_radius}_test_W{WEEKS}.json.gz'
        FINISHED_DATA_FILE2 = f'/Users/itlam/fomo/instances/Ryde/TD_res{resolution}_radius{ring_radius}_W{WEEKS}.json.gz'
        INIT_TARGET_FILE = f'/Users/itlam/fomo/instances/Ryde/initial_target_state_1066_res{resolution}_radius{ring_radius}.json'

        with gzip.open(FINISHED_DATA_FILE, 'rt', encoding='utf-8') as file:
            sim_data = json.load(file)
        
        with open(INIT_TARGET_FILE, 'r', encoding='utf-8') as f:
            init_data = json.load(f)

        sum_scooters = 0
        escooter_dict = {}
        for area in init_data["areas"]:
            area_id = area["id"]
            init_state = area["final_target_states"][0]

            escooter_dict[area_id] = [random.randint(40, 100) for i in range(init_state)]
            sum_scooters += init_state
        
        for area in sim_data["areas"]:
            area_id = area["id"]
            area["e_scooters"] = escooter_dict[area_id]
        
        with gzip.open(FINISHED_DATA_FILE2, 'wt', encoding='utf-8') as file:
            json.dump(sim_data, file, indent=4)
        
        print(resolution, ring_radius, sum_scooters)