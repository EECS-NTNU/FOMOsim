"""
SIM SETTINGS
"""

# Vehicle settings

VEHICLE_BATTERY_INVENTORY = 150 
VEHICLE_BIKE_INVENTORY = 20
VEHICLE_SPEED = 15
MINUTES_PER_ACTION = 0.5
MINUTES_CONSTANT_PER_ACTION = 1
SERVICE_TIME_FROM = 6 #06:00
SERVICE_TIME_TO = 20  #20=20:00

# Bike settings

BATTERY_LIMIT = 20.0 # Should be the same as BATTERY_LEVEL_LOWER_BOUND
BIKE_SPEED = 7 # default, normally calculated from input data
BATTERY_CHANGE_PER_MINUTE = 0.2

# Depot settings

DEFAULT_DEPOT_CAPACITY = 1000
CHARGE_TIME_PER_BATTERY = 60
SWAP_TIME_PER_BATTERY = 0.2
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
BATTERY_LEVEL_UPPER_BOUND = 40 #% will change battery 

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

# ------- PILOT PARAMETERS ----------

# SETTINGS_INSTANCE = 'BG_W35'
SETTINGS_INSTANCE = 'OS_W31'
# SETTINGS_INSTANCE = 'TD_W34_old'

# settings_state = "instances/ebike/"
settings_state = "instances/ebike_with_depot/"

settings_duration = 24*10
settings_num_vehicles = 2

settings_max_depth = 6 # best = 6
settings_number_of_successors = 3 # best = 3
settings_time_horizon = 40 # best = 40

settings_criticality_weights_sets = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.05, 0.9, 0.05, 0, 0, 0], [0.45, 0.1, 0.05, 0.2, 0.05, 0.15]] # best [[0.2, 0.15, 0.2, 0.15, 0.15, 0.15], [0.2, 0.4, 0.1, 0.05, 0.15, 0.1], [0.4, 0.1, 0.05, 0.2, 0.05, 0.2]]
# settings_criticality_weights_sets = [[0.2, 0.15, 0.2, 0.15, 0.15, 0.15], [0.2, 0.4, 0.1, 0.05, 0.15, 0.1], [0.4, 0.1, 0.05, 0.2, 0.05, 0.2]]

settings_evaluation_weights = [0.45, 0.1, 0.45] # best
settings_number_of_scenarios = 60 # best
settings_discounting_factor = 0.6 # best

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

# settings_list_of_factors = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
settings_list_of_instances = ['OS_W34', 'OS_W31', "NY_W31", "BO_W31",'BG_W35', 'TD_W34_old']

# settings_list_of_seeds=[10,11,12,13,14,15,16,17,18,19]
settings_list_of_seeds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# settings_list_of_seeds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
# settings_list_of_seeds = [2]

RESULT_FOLDER = str(SETTINGS_INSTANCE) + '_' + str(settings_num_vehicles) + 'V_' + str(len(settings_list_of_seeds)) +'S_' + str(settings_duration//24) + 'D_PILOT_' + ('T' if SORTED_BIKES else 'F') + ('T' if ONLY_SWAP_ALLOWED else 'F') + ('T' if USE_BATTERY_CRITICALITY else 'F')