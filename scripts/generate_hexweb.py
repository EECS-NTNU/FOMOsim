#MAIN
from hexweb import Hexagon, HexWeb
import json
import gzip
from json_settings import *

def generate_hex_json():
    # Init verdier til å generere hex web
    #lat, lng = 63.434440, 10.322995
    #lat, lng = 63.42544430016357, 10.442440513493388
    # Definerer bounding box koordinater
    bbox = (min_lon, max_lon, min_lat, max_lat)
    lat = (bbox[2] + bbox[3]) / 2
    lng = (bbox[0] + bbox[1]) / 2

    hex_web = HexWeb(lat, lng, resolution, ring_radius, total_scooters) 
    hex_web.generate_hex_web()

    # For å fjerne hex utenfor de geografiske grensene
    hex_web.hexagons = hex_web.filter_hexagons_outside_boundingbox(bbox)

    # Fjerner de som er under havet
    hex_web.hexagons = hex_web.filter_elevation_hex_centers(api_key)


    # Lagre hexagons som brukes
    new_json_hex_data = {
        'areas': [{'hex_id': hexagon.hex_id, 'location': hexagon.center, 'edges': hexagon.vertices, 'e_scooters': hexagon.e_scooters} for hexagon in hex_web.hexagons],
        "map": "TD_W34.png",
        "map_boundingbox": [
            min_lon, 
            max_lon, 
            min_lat, 
            max_lat
        ]
    }

    data = new_json_hex_data
    # Åpne filen i skrivemodus og bruk json.dump() for å skrive datastrukturen til filen
    with open(HEX_FILE_PATH, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)