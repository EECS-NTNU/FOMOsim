"""
SIM SETTINGS
"""

# Vehicle settings
VEHICLE_BATTERY_INVENTORY = 150 # How many batteries can a vehicle hold?
VEHICLE_BIKE_INVENTORY = 20 # How many bikes can be carried in a vehicle?
VEHICLE_SPEED = 15 # Average speed of a vehicle? (km/h)
MINUTES_PER_ACTION = 0.5 # Minutes to load/unload/battery swaps
MINUTES_CONSTANT_PER_ACTION = 1 # Constat time in addition (f.eks park the car and start again)
SERVICE_TIME_FROM = 6 # 06:00
SERVICE_TIME_TO = 20  # 20:00

# Bike settings
BATTERY_LIMIT_TO_USE = 20.0 # Battery limit for the bike to be up for rental
BATTERY_LIMIT_TO_SWAP = 70
BIKE_SPEED = 7 # Average speed of a bike
BATTERY_CHANGE_PER_MINUTE = 0.2 # Decrease in battery for each minute the bike is in use (1 = 1%, deflate after 100 minutes use)

# Depot settings
DEFAULT_DEPOT_CAPACITY = 1000 # How many bikes can be parked at depot?
CHARGE_TIME_PER_BATTERY = 60 # How long does it take for a battery to charge fully in minutes?
SWAP_TIME_PER_BATTERY = 0.2 # How many minutes does it take to change out the inventory for each battery?
CONSTANT_DEPOT_DURATION = 15

# Station settings
DEFAULT_STATION_CAPACITY = 20

# Other settings
FULL_TRIP = True
ITERATION_LENGTH_MINUTES = 60
STATION_CENTER_DELTA = 0
SIM_CACHE_DIR = "sim_cache"
TRAFFIC_LOGGING = False
WALKING_SPEED = 4
MAX_ROAMING_DISTANCE_SIMULATOR = 0.6 #km, for simulation
MAX_ROAMING_DISTANCE_SOLUTIONS = 0.35 #km, for decision making

# HLV
BATTERY_LEVEL_LOWER_BOUND = 20 #% not functionality if under

AVERAGE_LENGHT_OF_TRIP = 10 #minutes -> to calculate average_discount

OVERFLOW_CRITERIA = 2.1 # of target state
STARVATION_CRITERIA = 0.35 # of target state

BIKES_OVERFLOW_NEIGHBOR = 1
BIKES_STARVED_NEIGHBOR = 2

NEIGHBOR_BALANCE_PICKUP = 0.5
NEIGHBOR_BALANCE_DELIVERY = 0.5

SORTED_BIKES = True
ONLY_SWAP_ALLOWED = True
USE_BATTERY_CRITICALITY = True
FF_ROAMING_AREA_RADIUS = 3 # How many areas is a user willing to roam to find a bike?

# ------- PILOT PARAMETERS ----------

SB_INSTANCE_FILE = 'instances/TD_W34'
FF_INSTANCE_FILE = 'instances/Ryde/TD_W19_test_W3_NEW'
FF_TARGET_STATE_FILE = 'instances/Ryde/FINAL_target_state_1066_NEW.json.gz'
START_TIME = 7 # hour of day the simulation starts at
DURATION = 24*1
NUM_VEHICLES = 1

MAX_DEPTH = 1 # alpha, best = 6
NUM_SUCCESSORS = 1 # beta, best = 3
TIME_HORIZON = 30 # best = 40

CRITICAILITY_WEIGHTS_SET = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.05, 0.9, 0.05, 0, 0, 0], [0.45, 0.1, 0.05, 0.2, 0.05, 0.15]]
EVALUATION_WEIGHTS = [0.45, 0.1, 0.45]
NUM_SCENARIOS = 10
DISCOUNTING_FACTOR = 0.6

# CLUSTERING
MAX_WALKING_AREAS = 3
MAX_NUMBER_OF_CLUSTERS = 10
NEIGHBOR_BATTERY_LIMIT = 50
LOCATION_TYPE_MARGIN = 0.15

ADJUSTING_CRITICALITY = 0.5
RANDOM_DESTINATION_PROB = 0.02

# run_simulations.py
SEEDS_LIST=[10,11,12,13,14]

# RESULT_FOLDER = f'SEEDS{len(SEEDS_LIST)}_V{NUM_VEHICLES}_OR{MAX_WALKING_AREAS}_C{MAX_NUMBER_OF_CLUSTERS}_SCENARIOS{NUM_SCENARIOS}'
RESULT_FOLDER = ''