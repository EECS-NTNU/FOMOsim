"""
SIM SETTINGS
"""

# User interface settings

USER_INTERFACE_MODE = "GUI" # uses GUI from dashboard.py under tripStats
# USER_INTERFACE_MODE = "CMD" # uses normal code in main.py 
FULL_TRIP = True
TRAFFIC_LOGGING = False
REPORT_CHANGES = False # if True, changes in station position or name are reported in terminal

# Vehicle settings

VAN_BATTERY_INVENTORY = 170
VAN_SCOOTER_INVENTORY = 20
VEHICLE_SPEED = 15
MINUTES_PER_ACTION = 0.5
MINUTES_CONSTANT_PER_ACTION = 1

# Scooter settings
BATTERY_LIMIT = 20.0
SCOOTER_SPEED = 7 # default, normally calculated from input data
BATTERY_CHANGE_PER_MINUTE = 0.75

# Depot settings
MAIN_DEPOT_CAPACITY = 10000
SMALL_DEPOT_CAPACITY = 100
CHARGE_TIME_PER_BATTERY = 60
SWAP_TIME_PER_BATTERY = 0.4
CONSTANT_DEPOT_DURATION = 15

# Station settings
DEFAULT_STATION_CAPACITY = 20

# Other settings

ITERATION_LENGTH_MINUTES = 60
CLUSTER_CENTER_DELTA = 0
SIM_CACHE_DIR = "sim_cache"

