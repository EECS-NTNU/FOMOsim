year = 2023
month = 5
day = 8
hour = 8
WEEKS = 3

resolution = 11 #11, 10, 9 og 8 
ring_radius = 100 #100, 58, 22, 10
total_scooters = 1066
api_key = "AIzaSyCIfbXi27GhuCLy_xED7FvscFxHnl8OhRQ"

VEHICLE_SPEED = 25
BIKE_SPEED = 7

RYDE_FILE_PATH = f'/Users/isabellam/NTNU/H2023/Prosjektoppgave/fomo/instances/Ryde/trips_W{WEEKS}_test.json'
FINISHED_DATA_FILE = f'/Users/itlam/fomo/instances/Ryde/TD_res{resolution}_radius{ring_radius}_test_W{WEEKS}.json.gz'
FINISHED_DATA_FILE2 = f'/Users/isabellam/NTNU/H2023/Prosjektoppgave/fomo/instances/Ryde/TD_res{resolution}_radius{ring_radius}_W{WEEKS}.json.gz'
HEX_FILE_PATH = f'/Users/isabellam/NTNU/H2023/Prosjektoppgave/fomo/instances/Ryde/hex_json_res{resolution}_radius{ring_radius}.json'
FINISHED_TARGET_STATE_DATA_FILE = f'/Users/itlam/fomo/instances/Ryde/target_states_res{resolution}_radius{ring_radius}.json.gz'
INITIAL_ESCOOTERS_FILE = f'/Users/isabellam/NTNU/H2023/Prosjektoppgave/fomo/instances/Ryde/initial_escooter_position_{total_scooters}.json'
INIT_TARGET_STATE = '/Users/isabellam/NTNU/H2023/Prosjektoppgave/fomo/instances/Ryde/initial_target_state_1066.json'
RYDE_DATA_SEPT = '/Users/isabellam/NTNU/H2023/Prosjektoppgave/fomo/instances/Ryde/3_weeks_sept.json'
INIT_TARGET_FILE = f'/Users/itlam/fomo/instances/Ryde/initial_target_state_1066_res{resolution}_radius{ring_radius}.json'

min_lon, max_lon, min_lat, max_lat = 10.350317, 10.489613, 63.396206, 63.457983 #Trondheim