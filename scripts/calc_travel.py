import geopy.distance
import json
import gzip

def calculate_traveltime(speed, vehicle_speed, hexagons):
    hexagons = read_data('../instances/Ryde/TD_W19_test_W3.json.gz')

    traveltime_matrix = {}
    traveltime_matrix_vehicle = {}
    for i, hex1 in enumerate(hexagons):
        for hex2 in hexagons[i+1:]:  # Start from the next location to avoid duplicates
            distance = _distance_to(hex1, hex2)
            travel_time = (distance / speed) * 60  # Calculate travel time in minutes

            # Set travel time for both directions due to symmetry
            traveltime_matrix[str(hex1['id']) + "_" + str(hex2['id'])] = travel_time
            traveltime_matrix[str(hex2['id']) + "_" + str(hex1['id'])] = travel_time

            travel_time_vehicle = (distance / vehicle_speed) * 60
            traveltime_matrix_vehicle[str(hex1['id']) + "_" + str(hex2['id'])] = travel_time_vehicle
            traveltime_matrix_vehicle[str(hex2['id']) + "_" + str(hex1['id'])] = travel_time_vehicle

        if i % 1000 == 0 or i == len(hexagons)-1:
            file_path = f'../instances/Ryde/travel/matrix_{i}.json.gz'
            json_data = {
                "travel_matrix": traveltime_matrix,
                "travel_vehicle_matrix": traveltime_matrix_vehicle
            }
            with gzip.open(file_path, 'wt', encoding='utf-8') as file:
                json.dump(json_data, file, ensure_ascii=False, indent=4)
            traveltime_matrix = {}
            traveltime_matrix_vehicle = {}

    return traveltime_matrix, traveltime_matrix_vehicle

def _distance_to(hex1, hex2):
    lat1, lon1 = hex1['location']
    lat2, lon2 = hex2['location']
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).km

def read_data(file_path):
    with gzip.open(file_path, 'r') as f:
        hex_data = json.load(f)
    return hex_data["areas"]

def save_data(file_path, travel_matrix, traveltime_matrix_vehicle):
    json_data = {
        "travel_matrix": travel_matrix,
        "travel_vehicle_matrix": traveltime_matrix_vehicle
    }
    with gzip.open(file_path, 'wt', encoding='utf-8') as file:
            json.dump(json_data, file, ensure_ascii=False, indent=4)

# area_list = read_data('../instances/Ryde/TD_W19_test_W3.json.gz')
# print(len(area_list))
travel_matrix, traveltime_matrix_vehicle = calculate_traveltime(7, 15, None)
# save_data('../instances/Ryde/travel_matrix.json.gz', travel_matrix, traveltime_matrix_vehicle)
