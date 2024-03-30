import json
from hexweb import HexWeb
import gzip
from json_settings import *
import multiprocessing as mp

def generate_json_parallel():
    # Les inn Ryde-data fra fil
    with open(RYDE_FILE_PATH, 'r', encoding='utf-8') as f:
        ryde_data = json.load(f)

    # Legg til arrival og depature intensities

    with open(HEX_FILE_PATH, 'r', encoding='utf-8') as f:
        hex_data = json.load(f)

    hexagons = HexWeb.generate_hex_web_from_json(hex_data)
    hexweb = HexWeb(0, 0, 0, 0, total_scooters, hexagons)

    hexweb.init_move_probabilities()
    parallel_find_arrival_departure_intensities(hexweb, ryde_data)
    # hexweb.find_arrival_departure_intensities_parallel(ryde_data)
    hexweb.calc_average_arrival_depature()
    hexweb.calc_move_probabilities()

    hexweb.distribute_scooters_random()
    traveltime = {} # hexweb.calculate_traveltime(BIKE_SPEED)
    traveltime_vehicle = {} # hexweb.calculate_traveltime(VEHICLE_SPEED)

    # Opprett den nye JSON-strukturen
    json_data = {
        'name': 'Ryde',
        'city': 'Trondheim',
        'areas': [{'id': hexagon.hex_id,
                   'location': hexagon.center, 
                   'edges': hexagon.vertices, 
                   'e_scooters': hexagon.e_scooters, 
                   'arrival_intensities': hexagon.average_arrival_intensities, 
                   'depature_intensities': hexagon.average_depature_intensities, 
                   'move_probabilites': hexagon.move_probabilities_normalized} 
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
    
    with gzip.open(f'../instances/Ryde/Parallell_TD_W19_test_W{WEEKS}.json.gz', 'wt', encoding='utf-8') as file:
            json.dump(json_data, file, ensure_ascii=False, indent=4)

def process_chunk(trip_chunk, stats_instance: HexWeb):
    for trip in trip_chunk:
        start_coords = trip["route"]["features"][0]["geometry"]["coordinates"]
        end_coords = trip["route"]["features"][-1]["geometry"]["coordinates"]
        start_time = trip["start_time"]
        end_time = trip["end_time"]
        
        stats_instance.add_trip_to_stats(start_coords, end_coords, start_time, end_time)
        
def chunk_data(data, n):
    """Split the data into n chunks."""
    for i in range(0, len(data), n):
        yield data[i:i + n]

def parallel_find_arrival_departure_intensities(stats_instance: HexWeb, data):
    n_processes = 4
    pool = mp.Pool(n_processes)
    chunk_size = len(data['data']) // n_processes
    chunks = list(chunk_data(data['data'], chunk_size))
    
    # Assuming StatsInstance can be pickled or doesn't need to be shared across processes,
    # otherwise, consider using a Manager or shared data structures.
    pool.starmap(process_chunk, [(chunk, stats_instance) for chunk in chunks])
    
    pool.close()
    pool.join()

generate_json_parallel()