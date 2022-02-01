class HyperParameters:
    """
    Class inherited by World class. Sets all parameters for the instance.
    """

    def __init__(
        self,
        DISCOUNT_RATE=0.90,
        EPSILON=0.1,
        DIVIDE_GET_POSSIBLE_ACTIONS=2,
        WEIGHT_UPDATE_STEP_SIZE=0.00001,
        ANN_LEARNING_RATE=0.0001,
        TRACE_DECAY=0.9,
        VEHICLE_INVENTORY_STEP_SIZE=0.25,
        WEIGHT_INITIALIZATION_VALUE=0.10,
        NUMBER_OF_ROLLOUTS=100,
        LOST_TRIP_REWARD=-1,
        NUMBER_OF_NEIGHBOURS=3,
        NUMBER_OF_RANDOM_NEIGHBOURS=3,
        MODELS_TO_BE_SAVED=10,
        TRAINING_SHIFTS_BEFORE_SAVE=500,
        SHIFT_DURATION=960,
        NUMBER_OF_VANS=4,
        NUMBER_OF_BIKES=0,
        ANN_NETWORK_STRUCTURE=None,
        LOCATION_REPETITION=3,
        INITIAL_EPSILON=0.9,
        FINAL_EPSILON=0.0001,
        REPLAY_BUFFER_SIZE=500,
        DEPOT_REWARD=1,
        PICK_UP_REWARD=0.5,
    ):
        self.DISCOUNT_RATE = DISCOUNT_RATE  # From sutton 0.9-0.99
        self.EPSILON = EPSILON  # Probability of taking a random action
        self.DIVIDE_GET_POSSIBLE_ACTIONS = DIVIDE_GET_POSSIBLE_ACTIONS

        # VALUE FUNCTION PARAMETERS
        self.WEIGHT_UPDATE_STEP_SIZE = WEIGHT_UPDATE_STEP_SIZE
        self.ANN_LEARNING_RATE = ANN_LEARNING_RATE
        self.VEHICLE_INVENTORY_STEP_SIZE = VEHICLE_INVENTORY_STEP_SIZE
        self.WEIGHT_INITIALIZATION_VALUE = WEIGHT_INITIALIZATION_VALUE
        self.TRACE_DECAY = TRACE_DECAY

        # Default simulation constants
        self.NUMBER_OF_ROLLOUTS = NUMBER_OF_ROLLOUTS
        self.BATTERY_LIMIT = BATTERY_LIMIT

        # Negative reward for lost trips
        self.LOST_TRIP_REWARD = LOST_TRIP_REWARD

        # Testing parameters
        self.NUMBER_OF_NEIGHBOURS = (
            NUMBER_OF_NEIGHBOURS  # Zero neighbors removes neighbor filtering
        )
        self.NUMBER_OF_RANDOM_NEIGHBOURS = NUMBER_OF_RANDOM_NEIGHBOURS
        self.MODELS_TO_BE_SAVED = MODELS_TO_BE_SAVED
        self.TRAINING_SHIFTS_BEFORE_SAVE = TRAINING_SHIFTS_BEFORE_SAVE
        self.SHIFT_DURATION = SHIFT_DURATION
        self.NUMBER_OF_VANS = NUMBER_OF_VANS
        self.NUMBER_OF_BIKES = NUMBER_OF_BIKES

        self.ANN_NETWORK_STRUCTURE = (
            ANN_NETWORK_STRUCTURE if ANN_NETWORK_STRUCTURE else [100, 100, 100]
        )

        self.LOCATION_REPETITION = LOCATION_REPETITION

        self.INITIAL_EPSILON = INITIAL_EPSILON
        self.FINAL_EPSILON = FINAL_EPSILON
        self.DEPOT_REWARD = DEPOT_REWARD
        self.PICK_UP_REWARD = PICK_UP_REWARD

        self.REPLAY_BUFFER_SIZE = REPLAY_BUFFER_SIZE


"""
WORLD SETTINGS
"""
ITERATION_LENGTH_MINUTES = 20
BATTERY_LIMIT = 20.0
CASE_BASE_LENGTH = 1000

# GEOSPATIAL DATA
GEOSPATIAL_BOUND_NEW = (59.9040, 59.9547, 10.6478, 10.8095)
CLUSTER_CENTER_DELTA = 0.001

# Inventory of vehicle
VAN_BATTERY_INVENTORY = 170
VAN_SCOOTER_INVENTORY = 20
BIKE_BATTERY_INVENTORY = 12
BIKE_SCOOTER_INVENTORY = 0

# Operational parameters
MINUTES_PER_ACTION = 2
MINUTES_CONSTANT_PER_ACTION = 5

# Speed of service vehicles
VEHICLE_SPEED = 15
MINUTES_IN_HOUR = 60

# Speed of scooter ref - Fearnley at al. (2020, section 3.6)
SCOOTER_SPEED = 7

# Depot parameters
MAIN_DEPOT_LOCATION = (59.931794, 10.788314)
SMALL_DEPOT_LOCATIONS = [
    (59.908009, 10.741604),
    (59.944473, 10.748624),
    (59.944473, 10.748624),
]
MAIN_DEPOT_CAPACITY = 10000
SMALL_DEPOT_CAPACITY = 100
CHARGE_TIME_PER_BATTERY = 60
SWAP_TIME_PER_BATTERY = 0.4
CONSTANT_DEPOT_DURATION = 15

"""
IMPORTANT PATHS
"""

STATE_CACHE_DIR = "test_state_cache"
WORLD_CACHE_DIR = "world_cache"

# Test data directory
TEST_DATA_DIRECTORY = "test_data"
EXCEL_EXPORT_DIR = "computational_study"

"""
VISUALIZATION
"""
# Colors for visualizer
BLUE, GREEN, RED, BLACK, WHITE = "blue", "green", "red", "black", "white"
VEHICLE_COLORS = ["blue", "green", "black", "purple"]

ACTION_OFFSET = 0.018

COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#000000",
    "#207188",
    "#86aa6d",
    "#db494c",
    "#34393c",
    "#eba432",
    "#c2c1bc",
    "#A4847E",
    "#056672",
    "#D8FB27",
    "#BBE5D1",
    "#FEEB81",
    "#126027",
    "#7666E7",
    "#530788",
    "#A281ED",
    "#954701",
    "#B42760",
    "#F0E466",
    "#A32315",
    "#4886E8",
    "#117427",
    "#A3A66A",
    "#F124AC",
    "#4572BD",
    "#93EB5F",
    "#ECDCCD",
    "#48317F",
    "#DF8547",
    "#1DE961",
    "#5BD669",
    "#4FAA9B",
    "#937016",
    "#840FF6",
    "#3EAEFD",
    "#F6F34D",
    "#015133",
    "#59025B",
    "#F03B29",
    "#53A912",
    "#34058C",
    "#FA928D",
    "#3C70C3",
    "#AB9869",
    "#B6BD37",
    "#693C24",
    "#2588F7",
    "#54B006",
    "#6604CE",
    "#4A4329",
    "#0175B1",
    "#177982",
    "#544FAD",
    "#DD5409",
    "#583ED1",
    "#CD9D69",
    "#6B0BCE",
    "#D14B12",
    "#96725D",
    "#BB137F",
    "#7B53B5",
    "#BFFB24",
    "#F9D08F",
    "#CF03B8",
    "#A6F591",
    "#D7CFDB",
    "#2D4AD6",
    "#BC5286",
    "#6245C8",
    "#E40EB7",
    "#E2DA97",
    "#EE5089",
    "#CAF026",
    "#668981",
    "#8E424B",
    "#49633D",
    "#8A4CE4",
    "#827C33",
    "#35EFF2",
    "#325041",
    "#2BC23F",
    "#44857A",
    "#DA0043",
    "#87A43F",
    "#D4FCEC",
    "#9FD87C",
    "#0D36DF",
    "#241B73",
    "#524526",
    "#163F53",
    "#4C9B58",
    "#00F4DB",
    "#20054B",
    "#82026F",
    "#CA561D",
    "#F94B06",
    "#5CCBDB",
    "#8B6882",
    "#9C28B0",
    "#15357B",
    "#BB00F4",
    "#451918",
    "#B94AE1",
    "#698290",
    "#415697",
    "#61B95D",
    "#957BD8",
    "#01A1C5",
    "#69E54F",
    "#D40C21",
    "#08A810",
    "#05ECC3",
    "#8FA2B5",
    "#D45A2C",
    "#1689EA",
    "#7DD21F",
    "#A615B6",
    "#430E4C",
    "#557F16",
    "#68E3A4",
    "#E19180",
    "#8B0197",
    "#7314C4",
    "#A397DA",
    "#175ACE",
    "#6185AD",
    "#D981A8",
    "#984ED3",
    "#37FFF0",
    "#90BB50",
    "#A818B0",
    "#28F263",
    "#700EA8",
    "#5C0D3A",
    "#CAF06F",
    "#815F36",
    "#CCF509",
    "#21C91D",
    "#D09B45",
    "#282AF6",
    "#053525",
    "#0FAE75",
    "#213E02",
    "#1572AA",
    "#9D9A3A",
    "#1C1DA9",
    "#C6A728",
    "#0BE59B",
    "#272CAF",
    "#75BA93",
    "#E29981",
    "#45F101",
    "#D8BA19",
    "#BF7545",
    "#0F85B1",
    "#E6DC7B",
    "#6B6548",
    "#78B075",
    "#AFDF4D",
    "#D0BD94",
    "#C6F81B",
    "#27C209",
    "#3C6574",
    "#2CE0B3",
    "#9C6E06",
    "#53CECD",
    "#A5EC06",
    "#AA83D6",
    "#7705D2",
    "#806015",
    "#881E9E",
    "#617730",
    "#1F9ACF",
    "#8AE30F",
    "#D1E1B4",
    "#D924F6",
    "#5FE267",
    "#6BDDF2",
    "#5E40A5",
    "#9B1580",
    "#B6E49C",
    "#619C46",
    "#504BDE",
]
