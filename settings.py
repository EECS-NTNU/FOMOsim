"""
SIM SETTINGS
"""
# ------- PRE-DETERMINED PARAMETERS ----------

# Vehicle settings
VEHICLE_BATTERY_INVENTORY = 50 # How many batteries can a vehicle hold?
VEHICLE_BIKE_INVENTORY = 20 # How many bikes can be carried in a vehicle?
VEHICLE_SPEED = 25 # Average speed of a vehicle? (km/h)
MINUTES_PER_ACTION = 1 # Minutes to load/unload/battery swaps
MINUTES_CONSTANT_PER_ACTION = 1.5 # Constat time in addition (f.eks park the car and start again)

# Bike settings
BATTERY_LIMIT_TO_USE = 20.0 # Battery limit for the bike to be up for rental
BIKE_SPEED = 7 # Average speed of a bike
ESCOOTER_SPEED = 13.2 # Average speed of a bike
BATTERY_CHANGE_PER_MINUTE = 0.4 # Decrease in battery for each minute the bike is in use (1 = 1%, deflate after 100 minutes use)

# Depot settings
DEFAULT_DEPOT_CAPACITY = 1000 # How many bikes can be parked at depot?
CHARGE_TIME_PER_BATTERY = 60 # How long does it take for a battery to charge fully in minutes?
SWAP_TIME_PER_BATTERY = 0.8 # How many minutes does it take to change out the inventory for each battery?
CONSTANT_DEPOT_DURATION = 15

# Station settings
DEFAULT_STATION_CAPACITY = 20

# Policy settings
FULL_TRIP = True
ITERATION_LENGTH_MINUTES = 60
STATION_CENTER_DELTA = 0
SIM_CACHE_DIR = "sim_cache"
TRAFFIC_LOGGING = False
RANDOM_DESTINATION_PROB = 0.02

START_TIME = 7 # hour of day the simulation starts at
DURATION = 24*60*7
NUM_VEHICLES = 2

# User behaviour
WALKING_SPEED = 4
MAX_ROAMING_DISTANCE_SIMULATOR = 0.6 #km, for simulation
MAX_ROAMING_DISTANCE_SOLUTIONS = 0.35 #km, for decision making
FF_ROAMING_AREA_RADIUS = 500//50 # How many areas is a user willing to roam to find a bike?
AVERAGE_LENGHT_OF_TRIP = 15 #minutes -> to calculate average_discount

# File settings
SB_INSTANCE_FILE = 'instances/TD_W34'
FF_INSTANCE_FILE = 'instances/Ryde/TD_W19_test_W3_NEW2'
FF_TARGET_STATE_FILE = 'instances/Ryde/FINAL_target_state_1066_NEW.json.gz'

# ------- COMPUTATIONAL VARIABLES ----------

# Vehicle settings
SERVICE_TIME_FROM = 0 # 06:00
SERVICE_TIME_TO = 24 # 20:00

# Bike settings
BATTERY_LIMIT_TO_SWAP = 70

# Policy settings
CONGESTION_CRITERIA = 0.9
OVERFLOW_CRITERIA = 2.1 # of target state
STARVATION_CRITERIA = 0.35 # of target state
BIKES_OVERFLOW_NEIGHBOR = 1 # how many additional bikes to pick up for each "starved" neighbor
BIKES_STARVED_NEIGHBOR = 2 # how many additional bikes to let be for each "starved" neighbor
MAX_DEPTH = 3 # alpha, best = 6
NUM_SUCCESSORS = 3 # beta, best = 3
TIME_HORIZON = 40 # best = 40
CRITICAILITY_WEIGHTS_SET = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.05, 0.9, 0.05, 0, 0, 0], [0.45, 0.1, 0.05, 0.2, 0.05, 0.15]]
EVALUATION_WEIGHTS = [0.45, 0.1, 0.45]
NUM_SCENARIOS = 10
DISCOUNTING_FACTOR = 0.6
MAX_NUMBER_OF_CLUSTERS = 10
ADJUSTING_CRITICALITY = 0
CLUSTER_USE_NEIGHBOURS = True

# ------- MANAGERIAL INSIGHTS ----------
MAX_WALKING_AREAS = 300 // 50

# ------- TODO -------
NEIGHBOR_BALANCE_PICKUP = 0.5 # hva er dette?
NEIGHBOR_BALANCE_DELIVERY = 0.5 # hva er dette?
SORTED_BIKES = True # remove?
ONLY_SWAP_ALLOWED = True # remove?
USE_BATTERY_CRITICALITY = True # remove?
LOCATION_TYPE_MARGIN = 0.15
NEIGHBOR_BATTERY_LIMIT = 50

SEEDS_LIST=[10,11,12,13,14,15,16,17,18,19]

RESULT_FOLDER = ''
TEST = True