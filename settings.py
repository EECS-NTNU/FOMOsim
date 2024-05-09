"""
SIM SETTINGS
"""
# ------- PRE-DETERMINED PARAMETERS ----------

# Vehicle settings
VEHICLE_BATTERY_INVENTORY = 50 # How many batteries can a vehicle hold?
VEHICLE_BIKE_INVENTORY = 20 # How many bikes can be carried in a vehicle?
VEHICLE_SPEED = 25 # Average speed of a vehicle? (km/h)
MINUTES_PER_ACTION = 3 # Minutes to load/unload/battery swaps
MINUTES_CONSTANT_PER_ACTION = 5 # Constat time in addition (f.eks park the car and start again)

# Bike settings
BATTERY_LIMIT_TO_USE = 15 # Battery limit for the bike to be up for rental
BIKE_SPEED = 7 # Average speed of a bike
ESCOOTER_SPEED = 13.2 # Average speed of a bike

# Depot settings
DEFAULT_DEPOT_CAPACITY = 1000 # How many bikes can be parked at depot?
CHARGE_TIME_PER_BATTERY = 60 # How long does it take for a battery to charge fully in minutes?
SWAP_TIME_PER_BATTERY = 0.8 # How many minutes does it take to change out the inventory for each battery?
CONSTANT_DEPOT_DURATION = 15

# Simulator settings
ITERATION_LENGTH_MINUTES = 60
SIM_CACHE_DIR = "sim_cache"
TRAFFIC_LOGGING = False
RANDOM_DESTINATION_PROB = 0.02 # 2% probability of going to a random destination
FULL_TRIP = True
START_TIME = 7 # hour of day the simulation starts at
DURATION = 24*60*7

# Station settings
DEFAULT_STATION_CAPACITY = 20

# Policy settings
NUM_VEHICLES = 2
BATTERY_CHANGE_PER_MINUTE = 0.4 # Decrease in battery for each minute the bike is in use (1 = 1%, deflate after 100 minutes use)

# User behaviour
WALKING_SPEED = 4
MAX_ROAMING_DISTANCE_SIMULATOR = 0.6 #km, for simulation -> BRUKES IKKE
MAX_ROAMING_DISTANCE_SOLUTIONS = 0.35 #km, for decision making
FF_ROAMING_AREA_RADIUS = 2 # How many areas is a user willing to roam to find a bike? (Ca. 500m)
AVERAGE_LENGHT_OF_TRIP = 15 #minutes -> to calculate average_discount

# File settings
RESOLUTION = 10 #11, 10, 9, 8
RADIUS = 58 #100, 58, 22, 10
SB_INSTANCE_FILE = 'instances/TD_W34'
FF_INSTANCE_FILE = f'instances/Ryde/TD_700_res{RESOLUTION}_radius{RADIUS}_W3'
FF_TARGET_STATE_FILE = f'instances/Ryde/target_states_700_res{RESOLUTION}_radius{RADIUS}.json.gz'

# ------- COMPUTATIONAL VARIABLES ----------

# Vehicle settings
SERVICE_TIME_FROM = 0 # 06:00
SERVICE_TIME_TO = 24 # 20:00

# Bike settings
BATTERY_LIMIT_TO_SWAP = 70

# Policy settings
CONGESTION_CRITERIA = 0.80
OVERFLOW_CRITERIA = 2.1 # of target state
STARVATION_CRITERIA = 0.2 # of target state

MAX_DEPTH = 6 # alpha, best = 6
NUM_SUCCESSORS = 3 # beta, best = 3

TIME_HORIZON = 40 # best = 40

CRITICAILITY_WEIGHTS_SET = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.05, 0.9, 0.05, 0, 0, 0], [0.45, 0.1, 0.05, 0.2, 0.05, 0.15]]
CRITICAILITY_WEIGHTS_SET_FF = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.05, 0.9, 0.05, 0, 0, 0], [0.45, 0.1, 0.05, 0.2, 0.05, 0.15]]
CRITICAILITY_WEIGHTS_SET_SB = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.05, 0.9, 0.05, 0, 0, 0], [0.45, 0.1, 0.05, 0.2, 0.05, 0.15]]
ADJUSTING_CRITICALITY = 1

EVALUATION_WEIGHTS = [1/3, 1/3, 1/3]

NUM_SCENARIOS = 50

DISCOUNTING_FACTOR = 0.5

MAX_NUMBER_OF_CLUSTERS = 10

CLUSTER_USE_NEIGHBOURS = True

LOCATION_TYPE_MARGIN = 0.15

NEIGHBOR_BATTERY_LIMIT = 50

# ------- MANAGERIAL INSIGHTS ----------
OPERATOR_RADIUS = 1
BIKES_CONGESTED_NEIGHBOR = 1 # how many additional bikes to pick up for each "starved" neighbor
BIKES_STARVED_NEIGHBOR = 2 # how many additional bikes to let be for each "starved" neighbor

# ------- TODO -------

# Kan disse fjernes, brukes bare i prosjektoppgaven?
NEIGHBOR_BALANCE_PICKUP = 0.5 # hva er dette?
NEIGHBOR_BALANCE_DELIVERY = 0.5 # hva er dette?
ONLY_SWAP_ALLOWED = True
USE_BATTERY_CRITICALITY = True # remove?

SORTED_BIKES = True # remove? gjøre om i koden til at denne er true

SEEDS_LIST=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

RESULT_FOLDER = ''
TEST = False