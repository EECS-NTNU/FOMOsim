
from hexweb import Hexagon, HexWeb 
import json
from json_settings import *
import gzip



def generate_json_target_state():

    # with open(FINISHED_DATA_FILE, 'r', encoding='utf-8') as f:
    #     hex_data = json.load(f)

    hex_data ={
        'areas': [
            #  {
            #     "hex_id": "1",
            #     "arrival_intensities": [[0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] for _ in range(7)],
            #     "depature_intensities": [[0 for i in range(24)] for _ in range(7)],
            #  },
             {
                "hex_id": "2",
                "arrival_intensities": [[0 for _ in range(24)] for _ in range(7)],
                "depature_intensities": [[0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] for _ in range(7)],
             },
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
        ]
    }
    hexagons = HexWeb.generate_hex_web_target_state_from_json(hex_data)

    for hex in hexagons:
        hex.initial_target_states = 100 #finne en måte å gjøre dette på 
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

    with open(FINISHED_TARGET_STATE_DATA_FILE, 'w', encoding='utf-8') as file:
            json.dump(json_target_state_data, file, ensure_ascii=False, indent=4)

generate_json_target_state()
