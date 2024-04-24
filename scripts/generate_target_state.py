
from hexweb import Hexagon, HexWeb 
import json
from json_settings import *
import gzip



def generate_json_target_state():
    
    FINISHED_DATA_FILE = '/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/TD_W19_test_W3_NEW.json.gz'
    INIT_TARGET_STATE = '/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/initial_target_state_1066.json'

    with gzip.open(FINISHED_DATA_FILE, 'rt', encoding='utf-8') as f:
        hex_data = json.load(f)

    # sumlist = {i: 0 for i in range(24)}
    # for area in hex_data['areas']:
    #     for day in range(7):
    #         for hour in range(24):
    #             sumlist[hour] += area['arrival_intensities'][day][hour]
    # print(f'sumlist = {sumlist}')
    # return
        
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
    
    #distribute initial escooter positions based on initial target states
    #hexweb.distribute_scooters_target_state()

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

    FINISHED_TARGET_FILE = "/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/FINAL_target_state_1066_NEW_ceil.json.gz"
    # Lagre data som en gzip-komprimert JSON-fil
    with gzip.open(FINISHED_TARGET_FILE, 'wt', encoding='ascii') as zipfile:
        json.dump(json_target_state_data, zipfile)
    
    with open('/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/FINAL_target_state_1066_NEW_ceil.json', 'w', encoding='utf-8') as f:
        json.dump(json_target_state_data, f, indent=4)
    


generate_json_target_state()
        
with open('/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/FINAL_target_state_1066_NEW_ceil.json', 'r') as file:
    data = json.load(file)

counter = 0 
final_target_states = [[0 for j in range(24)] for i in range(7)]
hex_data_dict = {}
targert_states = [[0 for i in range(24)] for j in range(7)]
     
for hex_data in data['areas']:  
    
    hex_final_target_states = hex_data['target_states']
    for i in range(7):
        for j in range(24):
            final_target_states[i][j] += hex_final_target_states[i][j]

with open('/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/check_FINAL_states_ceil.json', 'w', encoding='utf-8') as f:
        json.dump(final_target_states, f, indent=4)