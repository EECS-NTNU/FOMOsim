"""
SIM SETTINGS
"""

# Vehicle settings

VEHICLE_BATTERY_INVENTORY = 170
VEHICLE_BIKE_INVENTORY = 20
VEHICLE_SPEED = 15
MINUTES_PER_ACTION = 0.5
MINUTES_CONSTANT_PER_ACTION = 1
SERVICE_TIME_FROM = 6 #06:00
SERVICE_TIME_TO = 20  #20=20:00

# Bike settings

BATTERY_LIMIT = 20.0
BIKE_SPEED = 7 # default, normally calculated from input data
BATTERY_CHANGE_PER_MINUTE = 0.75

# Depot settings

DEFAULT_DEPOT_CAPACITY = 1000
CHARGE_TIME_PER_BATTERY = 60
SWAP_TIME_PER_BATTERY = 0.4
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