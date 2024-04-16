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
FF_ROAMING_AREA_RADIUS = 3

# ------- PILOT PARAMETERS ----------

# SETTINGS_INSTANCE = 'BG_W35'
# INSTANCE_FILE = 'OS_W31'
INSTANCE_FILE = 'TD_W34'

# settings_state = "instances/ebike/"
# STATE_FOLDER = "instances/ebike_with_depot/"

DURATION = 24*1
NUM_VEHICLES = 1

MAX_DEPTH = 1 # best = 6
NUM_SUCCESSORS = 1 # best = 3
TIME_HORIZON = 30 # best = 40

CRITICAILITY_WEIGHTS_SET = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.05, 0.9, 0.05, 0, 0, 0], [0.45, 0.1, 0.05, 0.2, 0.05, 0.15]] # best [[0.2, 0.15, 0.2, 0.15, 0.15, 0.15], [0.2, 0.4, 0.1, 0.05, 0.15, 0.1], [0.4, 0.1, 0.05, 0.2, 0.05, 0.2]]
# settings_criticality_weights_sets = [[0.2, 0.15, 0.2, 0.15, 0.15, 0.15], [0.2, 0.4, 0.1, 0.05, 0.15, 0.1], [0.4, 0.1, 0.05, 0.2, 0.05, 0.2]]

EVALUATION_WEIGHTS = [0.45, 0.1, 0.45] # best
NUM_SCENARIOS = 10 # best
DISCOUNTING_FACTOR = 0.6 # best

# settings_list_of_timehorizons = [10,20,30,40,50,60]
# settings_evaluation_weights = dict(
#     a = [0.4, 0.3, 0.3], b=[0.8, 0.1, 0.1], c=[0.1, 0.8, 0.1], d=[0.1, 0.1, 0.8],
#     e=[0.6, 0.1, 0.3], f=[0.3, 0.6, 0.1], g=[0.3, 0.1, 0.6], h=[0.6, 0.3, 0.1],
#     i=[1.0, 0.0, 0.0], j=[0.45, 0.45, 0.1], k=[0.45, 0.1, 0.45], l=[0.33, 0.33, 0.33],
#     m=[0.9, 0.05, 0.05], n=[0.95, 0.04, 0.01], o=[0.85, 0.1, 0.05], p=[0.9, 0.09, 0.01],
#     q=[0.5, 0.05, 0.45], r=[0.45, 0.05, 0.5], s=[0.5, 0, 0.5], t=[0.4, 0.2, 0.4] <- tester siden k var best i første omgang
#     )
# settings_criticality_weights = dict(
#     a=[[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]], b=[[0.3, 0.15, 0.25, 0.2, 0.1, 0]], c=[[0.15, 0.3, 0.15, 0.1, 0.1, 0.2]], d=[[0.25, 0.25, 0.1, 0.1, 0.2, 0.1]], # Balanced
#     e=[[0.15, 0.4, 0.05, 0.05, 0, 0.35]], f=[[0.05, 0.9, 0.05, 0, 0, 0]], g=[[0.1, 0.5, 0.1, 0.1, 0.1, 0.1]], h=[[0.3, 0.5, 0, 0, 0.2, 0]], # Long term
#     i=[[0.4, 0, 0, 0.1, 0, 0.5]], j=[[0.6, 0.05, 0.05, 0.1, 0.05, 0.15]], k=[[0.6, 0.1, 0.05, 0.2, 0.05, 0]], l=[[0.5, 0.05, 0.1, 0.05, 0.1, 0.2]], # Short term
#     m=[[1, 0, 0, 0, 0, 0]], n=[[0, 1, 0, 0, 0, 0]], o=[[0, 0, 1, 0, 0, 0]], # Extremes
#     p=[[0, 0, 0, 1, 0, 0]], q=[[0, 0, 0, 0, 1, 0]], r=[[0, 0, 0, 0, 0, 1]], # Extremes

#     s= [[0.05, 0.85, 0.05, 0, 0, 0.05]], # f
#     t=[[0.25, 0.45, 0, 0, 0.2, 0.1]], # h
#     u = [[0.3, 0.45, 0, 0, 0.2, 0.05]], # h
#     v =[[0.5, 0.1, 0.05, 0.2, 0.05, 0.1]], # k
#     w =[[0.55, 0.1, 0.05, 0.2, 0.05, 0.05]], # k
#     x =[[0.45, 0.1, 0.05, 0.2, 0.05, 0.15]], # k
#     y =[[0.6, 0.1, 0.05, 0.15, 0.05, 0.05]], # k
#     z =[[0.6, 0.05, 0.05, 0.15, 0.05, 0.1]] # k
#     # Long term
# )

# Testing parameters
DISCOUNTING_FACTOR_LIST = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
INSTANCES_LIST = ['OS_W34', 'OS_W31', "NY_W31", "BO_W31",'BG_W35', 'TD_W34_old']
SEEDS_LIST=[10,11,12,13,14,15,16,17,18,19]

RESULT_FOLDER = 'Collab2_FF' + str(INSTANCE_FILE) + '_' + str(NUM_VEHICLES) + 'V_' + str(len(SEEDS_LIST)) +'S_' + str(DURATION//24) + 'D_PILOT_' + ('T' if SORTED_BIKES else 'F') + ('T' if ONLY_SWAP_ALLOWED else 'F') + ('T' if USE_BATTERY_CRITICALITY else 'F')

# CLUSTERING
MAX_WALKING_AREAS = 3
MAX_NUMBER_OF_CLUSTERS = 10
NEIGHBOR_BATTERY_LIMIT = 50
LOCATION_TYPE_MARGIN = 0.15

ADJUSTING_CRITICALITY = 0.5
RANDOM_DESTINATION_PROB = 0.02