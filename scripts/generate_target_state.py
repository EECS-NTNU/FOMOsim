
from hexweb import Hexagon, HexWeb 
import json
from json_settings import *
import gzip



def generate_json_target_state():

     

    #hex_data ={
        #'areas': [
            #  {
            #     "hex_id": "1",
            #     "arrival_intensities": [[0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] for _ in range(7)],
            #     "depature_intensities": [[0 for i in range(24)] for _ in range(7)],
            #  },
            # {
            #    "hex_id": "2",
            #    "arrival_intensities": [[0 for _ in range(24)] for _ in range(7)],
            #    "depature_intensities": [[0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] for _ in range(7)],
            # },
            #  {
            #     "hex_id": "3",
            #     "arrival_intensities": [[0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] for _ in range(7)],
            #     "depature_intensities": [[0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,1,1,1,1,1,1,1,1,1] for _ in range(7)],
            #  },
            #  {
            #     "hex_id": "4",
            #     "arrival_intensities": [[0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] for _ in range(7)],
            #     "depature_intensities": [[0,0,0,0,0,0,0,0,0,0,0,0,2,2,1,2,1,1,1,1,1,1,1,1] for _ in range(7)],
            #  },
            #  {
            #     "hex_id": "5",
            #     "arrival_intensities": [[0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] for _ in range(7)],
            #     "depature_intensities": [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12] for _ in range(7)],
            #  },
        #]
   # }
    
    FINISHED_DATA_FILE = "/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/TD_W19_test_W3.json.gz"
    INIT_TARGET_STATE = "/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/initial_target_state_1066.json"

    with gzip.open(FINISHED_DATA_FILE, 'rt', encoding='utf-8') as f:
        hex_data = json.load(f)
    
    with open(INIT_TARGET_STATE, 'r', encoding='utf-8') as f:
         initial_target_state_data = json.load(f)

    hexagons = HexWeb.generate_hex_web_target_state_from_json(hex_data)
   #HexWeb.set_initial_target_state(hexagons, initial_target_state_data)
    #def set_initial_target_state(hexagons, initial_target_state_data):
    for init_hex in initial_target_state_data['areas']:
        for hexagon in hexagons:
            if hexagon.hex_id == init_hex['id']:
                hexagon.initial_target_states = init_hex['final_target_states']
                break

    for hex in hexagons:
        hex.calc_target_state()
    #hexweb.calc_move_probabilities()

    #hexweb.distribute_scooters_random()

    json_target_state_data = {
        'name': 'Ryde',
        'city': 'Trondheim',
        'areas': [{'id': hexagon.hex_id,'target_states': hexagon.target_states} for hexagon in hexagons]
    }
    # print(hex_web.count_trips, hex_web.count_same_start_end_hex,hex_web.no_start_hex, hex_web.no_end_hex)

    # with gzip.open(write_file, 'wt', encoding="ascii") as zipfile:
    #        json.dump(json_data, zipfile)

    FINISHED_TARGET_STATE_DATA_FILE = "/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/finished_target_state_1066.json.gz"
    # Lagre data som en gzip-komprimert JSON-fil
    with gzip.open(FINISHED_TARGET_STATE_DATA_FILE, 'wt', encoding='ascii') as zipfile:
        json.dump(json_target_state_data, zipfile)
    
    with open("/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/finished_target_state_1066_json.json", 'w', encoding='utf-8') as f:
        json.dump(json_target_state_data, f, indent=4)

generate_json_target_state()
