import json
from hexweb import Hexagon, HexWeb 
import gzip
from json_settings import *

def generate_json():
    # Les inn Ryde-data fra fil
    with open(RYDE_FILE_PATH, 'r', encoding='utf-8') as f:
        ryde_data = json.load(f)

    # Legg til arrival og depature intensities

    with open(HEX_FILE_PATH, 'r', encoding='utf-8') as f:
        hex_data = json.load(f)
    hexagons = HexWeb.generate_hex_web_from_json(hex_data)

    hexweb = HexWeb(0, 0, 0, 0, total_scooters, hexagons)

    hexweb.init_move_probabilities()
    hexweb.find_arrival_depature_intensities(ryde_data)

    hexweb.calc_average_arrival_depature()
    hexweb.calc_move_probabilities()

    hexweb.distribute_scooters_random()

    # Opprett den nye JSON-strukturen
    json_data = {
        'name': 'Ryde',
        'city': 'Trondheim',
        'areas': [{'id': hexagon.hex_id,'location': hexagon.center, 'edges': hexagon.vertices, 'e_scooters': hexagon.e_scooters, 'arrival_intensities': hexagon.average_arrival_intensities, 'depature_intensities': hexagon.average_depature_intensities, 'move_probabilites': hexagon.move_probabilities_normalized} for hexagon in hexweb.hexagons],
        "map_boundingbox": [
            min_lon, 
            max_lon, 
            min_lat, 
            max_lat
        ]
        #'map': {'same_values_as_before': True}  # Placeholder for map-related values
    }
    # print(hex_web.count_trips, hex_web.count_same_start_end_hex,hex_web.no_start_hex, hex_web.no_end_hex)

    # with gzip.open(write_file, 'wt', encoding="ascii") as zipfile:
    #        json.dump(json_data, zipfile)

    with gzip.open(FINISHED_DATA_FILE, 'wt', encoding='utf-8') as file:
            json.dump(json_data, file, ensure_ascii=False, indent=4)