"""
SIM SETTINGS
"""

# Vehicle settings

VEHICLE_BATTERY_INVENTORY = 170
VEHICLE_SCOOTER_INVENTORY = 20
VEHICLE_SPEED = 15
MINUTES_PER_ACTION = 0.5
MINUTES_CONSTANT_PER_ACTION = 1

HANDLING_TIME = 0.25  #these are not SIM settings, but for gleditsch and hagen -> Move
PARKING_TIME = 2 #these are not SIM settings, but for gleditsch and hagen -> Move

# Scooter settings
BATTERY_LIMIT = 20.0
SCOOTER_SPEED = 7 # default, normally calculated from input data
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
