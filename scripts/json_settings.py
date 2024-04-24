year = 2023
month = 5
day = 8
hour = 8
WEEKS = 3

resolution = 10 #11, 10, 9 og 8 
ring_radius = 58 #100, 58, 22, 10
total_scooters = 1066
api_key = "AIzaSyCIfbXi27GhuCLy_xED7FvscFxHnl8OhRQ"

VEHICLE_SPEED = 25
BIKE_SPEED = 7

RYDE_FILE_PATH = f'../instances/Ryde/trips_W{WEEKS}_test.json' # '../instances/Ryde/Ryde_data_trips_may_1_days.json'
FINISHED_DATA_FILE = f'../instances/Ryde/TD_res{resolution}_radius{ring_radius}_test_W{WEEKS}.json.gz'
HEX_FILE_PATH = f'../instances/Ryde/hex_json_res{resolution}_radius{ring_radius}.json'
FINISHED_TARGET_STATE_DATA_FILE = f'../instances/Ryde/target_states_res{resolution}_radius{ring_radius}.json.gz'
INITIAL_ESCOOTERS_FILE = f'../instances/Ryde/initial_escooter_position_{total_scooters}.json'

min_lon, max_lon, min_lat, max_lat = 10.350317, 10.489613, 63.396206, 63.457983 #Trondheim