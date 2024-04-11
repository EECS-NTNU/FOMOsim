year = 2023
month = 5
day = 8
hour = 8
WEEKS = 3

resolution = 11
ring_radius = 100
total_scooters = 10
api_key = "AIzaSyCu-x049Ut84I6xTMojTJrAK1z5QZ49D9E"

VEHICLE_SPEED = 30
BIKE_SPEED = 8

RYDE_FILE_PATH = f'../instances/Ryde/trips_W{WEEKS}_test.json' # '../instances/Ryde/Ryde_data_trips_may_1_days.json'
FINISHED_DATA_FILE = f'../instances/Ryde/TD_W19_test_W{WEEKS}.json.gz'
HEX_FILE_PATH = f'../instances/Ryde/hex_json_res{resolution}_radius{ring_radius}.json'
FINISHED_TARGET_STATE_DATA_FILE = f'../instances/Ryde/target_states_TD_W19_test_W{WEEKS}.json.gz'

min_lon, max_lon, min_lat, max_lat = 10.350317, 10.489613, 63.396206, 63.457983 #Trondheim