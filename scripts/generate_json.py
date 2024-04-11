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
    traveltime = {} # hexweb.calculate_traveltime(BIKE_SPEED)
    traveltime_vehicle = {} # hexweb.calculate_traveltime(VEHICLE_SPEED)

    # Opprett den nye JSON-strukturen
    json_data = {
        'name': 'Ryde',
        'city': 'Trondheim',
        'areas': [{'id': hexagon.hex_id, # A0
                    'location': hexagon.center, # [lat, lon]
                    'edges': hexagon.vertices, # [(lat, lon) * 6]
                    'e_scooters': hexagon.e_scooters, # [86, 35, ...]
                    'arrival_intensities': hexagon.average_arrival_intensities, # [[x * 24] * 7]
                    'depature_intensities': hexagon.average_depature_intensities, # [[x * 24] * 7]
                    'move_probabilites': hexagon.move_probabilities_normalized} # [[{id: value, ...} * 24] * 7]
                    for hexagon in hexweb.hexagons],
        "map_boundingbox": [
            min_lon, 
            max_lon, 
            min_lat, 
            max_lat
        ],
        'depots': [{'id': "D0", 'location': [63.431711, 10.403537]}], #TODO noe mer vi trenger her?
        'traveltime': traveltime,
        'traveltime_stdev': {key: 0 for key in traveltime.keys()},
        'traveltime_vehicle': traveltime_vehicle,
        'traveltime_vehicle_stdev': {key: 0 for key in traveltime_vehicle.keys()}
    }

    with gzip.open(FINISHED_DATA_FILE, 'wt', encoding='utf-8') as file:
            json.dump(json_data, file, ensure_ascii=False, indent=4)