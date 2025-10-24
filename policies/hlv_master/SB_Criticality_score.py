from .Simple_calculations import calculate_net_demand, calculate_hourly_discharge_rate
from settings import *
# from .Variables import *

"""
Calculates the criticality score for a location to be visited by a vehicle.
Criticality score is based on:
- Time to violation = How long is it until a starvation/congestion appears
- Deviation from target state = The difference from the locations inventory and the current target state
- Neighborhood criticality = How much can neighboring locations affect the current locations criticality
- Demand = How the current demand contributes to the criticality
- Driving time = The time it takes for the vehicle to arrive contributes to the criticality
"""

def calculate_criticality(weights, state, potential_stations, station, total_num_bikes_in_system, visited_stations = None):
    """
    Returns a dictionary with station and criticality score for all potential station to visit. The higher the score the more critical the station is.

    Parameters:
    - weights = list of the weights of how much each component matters. Float from 0-1, summed to 1
    - state = State
    - potential_stations = list of Station-objects to consider visiting
    - total_num_bikes_in_system = int, number of bikes total
    - visited_stations = tabu_list og stations that should not be visited
    """
    [w_t, w_dev, w_n, w_dem, w_dri, w_bc] = weights
    criticalities = dict() # key: station, value: list of criticality scores for each category
    criticalities_summed = dict() # key: station, value: criticality of station

    # Lists of criticality scores for all potential_stations
    time_to_violation_list = []
    deviation_list = []
    neighborhood_crit_list = [] 
    demand_crit_list = []
    driving_time_list = [] 
    BL_composition_list = []

    for potential_station in potential_stations:
        net_demand = calculate_net_demand(potential_station, state.time, state.day(), state.hour(), TIME_HORIZON)
        target_state = potential_station.get_target_state(state.day(),state.hour())
        potential_station_type = calculate_station_type(target_state, len(potential_station.get_available_bikes()) + net_demand)

        # Time to violation
        time_to_violation = calculate_time_to_violation(net_demand, potential_station, state, total_num_bikes_in_system)
        time_to_violation_list.append(time_to_violation)

        # Deviation from target state
        deviation_from_target_state = calculate_deviation_from_target_state(potential_station, net_demand, target_state)
        deviation_list.append(deviation_from_target_state)

        # Neighborhood criticality
        neighborhood_crit = calculate_neighborhood_criticality(state, potential_station, potential_station_type, visited_stations)
        neighborhood_crit_list.append(neighborhood_crit)

        # Demand criticality
        demand_crit = calculate_demand_criticality(potential_station_type, net_demand)
        demand_crit_list.append(demand_crit)

        # Driving time criticality
        driving_time_crit = calculate_driving_time_crit(state, station, potential_station)
        driving_time_list.append(driving_time_crit)

        # Battery level composition criticality
        battery_level_comp_crit = calculate_battery_level_composition_criticality(state, potential_station, total_num_bikes_in_system)
        BL_composition_list.append(battery_level_comp_crit)

        criticalities[potential_station] = [time_to_violation, deviation_from_target_state, neighborhood_crit, demand_crit, driving_time_crit, battery_level_comp_crit]
    
    criticalities_normalized = normalize_results(criticalities, time_to_violation_list, deviation_list, neighborhood_crit_list, demand_crit_list, driving_time_list, BL_composition_list)

    for station in criticalities_normalized:
        criticalities_normalized[station][0] *= w_t
        criticalities_normalized[station][1] *= w_dev
        criticalities_normalized[station][2] *= w_n
        criticalities_normalized[station][3] *= w_dem
        criticalities_normalized[station][4] *= w_dri
        criticalities_normalized[station][5] *= w_bc

        criticalities_summed[station] = criticalities_normalized[station][0] + criticalities_normalized[station][1] 
        + criticalities_normalized[station][2] + criticalities_normalized[station][3] + criticalities_normalized[station][4] + criticalities_normalized[station][5]

    #Sorted in descending order - most critical first
    criticalities_summed = dict(sorted(criticalities_summed.items(), key=lambda item: item[1], reverse=True))

    return criticalities_summed

def calculate_time_to_violation(net_demand, station, state, total_num_bikes_in_system):
    """
    Returns a float-score of how critical the station is based on how many hours until a starvation/congestion occurs.
    A low score is critical, while high is less critical

    Parameters:
    - net_demand = float, the net demand calculated from the location's arrive/leave intensities (positive = bikes are arriving, negative = bikes are departuring)
    - station = Station-object under consideration
    - state = State
    - total_num_bikes_in_system = total number of bikes in the system
    """
    time_to_violation = 0
    
    # Calculate if a congestion might occur, no battery is needed in this calculation as we only care if you are not able to park
    if net_demand > 0:
        remaining_station_capacity = station.capacity - station.number_of_bikes()
        time_to_violation = remaining_station_capacity / net_demand

    # Calculate time if a starvation might occur
    elif net_demand < 0:
        sorted_bikes_by_battery = sorted(station.bikes.values(), key=lambda bike: bike.battery, reverse=False)

        # Time until there is no bikes left at a station that has sufficiant battery level
        violation_demand = len(station.get_available_bikes()) / -net_demand

        # Calculate the time until there are only bikes at station with too low battery for rental, based on hourly discharge in system
        bikes_most_charged = [bike.battery for bike in sorted_bikes_by_battery[-3:]]
        average_battery_top3 = sum(bikes_most_charged)/len(bikes_most_charged) if len(bikes_most_charged) > 0 else 0
        battery_over_limit_top3 = max(average_battery_top3 - BATTERY_LIMIT_TO_USE, 0)
        hourly_discharge = calculate_hourly_discharge_rate(state, total_num_bikes_in_system, False, True)
        if hourly_discharge == 0:
            rate = calculate_hourly_discharge_rate(state, total_num_bikes_in_system, False, True)
        violation_battery = battery_over_limit_top3 / hourly_discharge

        time_to_violation = min(
            violation_demand, 
            violation_battery,
            8
        )
    
    # Set time to violation far in the future
    else:
        time_to_violation = 8
    
    return time_to_violation

def calculate_deviation_from_target_state(station, net_demand, target_state):
    """
    Returns float-score for the deviation criticality.
    The higher the score, thus more critical.

    Parameters:
    - station = Station-object under consideration
    - net_demand = expected demand for the station during the time horizon
    - target_state = number of bikes ideally to be at the station
    """
    
    available_bikes_future = len(station.get_available_bikes()) + net_demand

    # If there is expected to still be bikes at station find the difference
    if available_bikes_future > 0:
        deviation_from_target_state = abs(target_state - available_bikes_future)
    else:
        deviation_from_target_state = target_state
    
    return deviation_from_target_state

def calculate_neighborhood_criticality(state, station, station_type, visited_stations):
    """
    Returns a int-score on how the neighboring stations, impact the criticality.
    The higher the score, the more critical.

    Parameters:
    - state = State
    - station = Station-object under consideration
    - station_type = if it is a pick up or delivery station (or balanced)
    - vistited_stations = list of Station-objects that are in the tabu_list
    """
    neighborhood_crit = 0
    neighbors = station.neighbours
    
    for neighbor in neighbors:
        station_crit = 0

        # If this neighoring station has been visited, it is not critical
        if visited_stations != None and neighbor.id in visited_stations:
            station_crit -= 3

        # If it has not been visited
        else:
            neighbor_net_demand = calculate_net_demand(neighbor, state.time, state.day(), state.hour(), TIME_HORIZON)
            neighbor_target_state = neighbor.get_target_state(state.day(),state.hour())

            expected_number_usable_bikes = len(neighbor.get_available_bikes()) + neighbor_net_demand
            neighbor_type = calculate_station_type(neighbor_target_state, expected_number_usable_bikes)

            # Same status increases the criticality
            if neighbor_type == station_type:
                station_crit += 1

                # Demand of neighbors adjusts the criticality
                station_crit += calculate_demand_criticality(neighbor_type, neighbor_net_demand)

            expected_number_bikes_neighbor = neighbor.number_of_bikes() + neighbor_net_demand

            # Decreases the criticality if there are complementary station types
            if station_type == 'p' and neighbor.capacity - expected_number_bikes_neighbor > 0: # The neighbor can help with arrivals
                station_crit -= 1
            elif station_type == 'd' and expected_number_usable_bikes > 0: # The neighbor can help with departures
                station_crit -= 1
            elif station_type == 'b':
                station_crit -= 1
            
            #Battery level composition
            # if station_type == 'd':
            #     battery_levels_neighbor = [bike.battery for bike in neighbor.get_available_bikes()]
            #     avg_battery_level = (sum(battery_levels_neighbor) / len(battery_levels_neighbor)) if len(battery_levels_neighbor) > 0 else 0

            #     if neighbor_type == 'd':
            #         if avg_battery_level < NEIGHBOR_BATTERY_LIMIT:
            #             station_crit += 1
            
            
        # Calculate the distance between the neighbor and current station, use this to adjust the station's criticality
        distance = station.distance_to(*neighbor.get_location())
        station_crit *= (1 - (distance/MAX_ROAMING_DISTANCE_SOLUTIONS))

        neighborhood_crit += station_crit
    
    return neighborhood_crit

def calculate_demand_criticality(station_type, net_demand):
    """
    Returns the criticality score based on demand and station type.
    The higher the score, the more critical.

    Parameters:
    - station_type = character describing if the station under consideration is a pick up, delivery or balanced
    - net_demand = how the flow at the station is for the set time_horizon
    """

    # The station has too many bikes and more are arriving -> positive
    if station_type == "p" and net_demand > 0:
        demand_crit = net_demand
    
    # The station has too many bikes and bikes are departuring -> positive
    elif station_type == "p" and net_demand < 0:
        demand_crit = -net_demand
    
    # The station had too few bikes but bikes are arriving here -> negative
    elif station_type == "d" and net_demand > 0:
        demand_crit = -net_demand
    
    # The station had too few bikes, and bikes are departuring -> negative
    elif station_type == "d" and net_demand < 0:
        demand_crit = net_demand
    
    # Station is balanced or no demand -> not critical
    else:
        demand_crit = 0
    
    return demand_crit

def calculate_driving_time_crit(state, current_station, potential_station):
    """
    Returns the driving time between two stations.
    The lower the score the more critical.
    """
    return state.get_vehicle_travel_time(current_station.id, potential_station.id)

def calculate_battery_level_composition_criticality(state, station, total_num_bikes_in_system):
    """
    Returns a float-score based on how the battery levels may be in the future.
    The lower the score, the more critical.

    Parameters:
    - state = State
    - station = Station-object under consideration
    - total_num_bikes_in_system = total number of bikes in the system
    """
    return 0
    current_escooters = station.bikes
    hourly_discharge_rate = calculate_hourly_discharge_rate(state, total_num_bikes_in_system, False, True)

    # Make list of the batteries of usable bikes at the station in current time and the next hour 
    battery_levels_current = [escooter.battery for escooter in current_escooters.values() if escooter.usable()]
    battery_levels_after = [escooter.battery - hourly_discharge_rate for escooter in current_escooters.values() if (escooter.battery - hourly_discharge_rate) > BATTERY_LIMIT_TO_USE]

    # Very critical if there are no bikes with sufficient battery left
    if len(battery_levels_after) == 0 or len(battery_levels_current) == 0:
        # If it is expected to "loose" all bikes with battery -> very critical
        if len(battery_levels_after) < len(battery_levels_current):
            return 0
        # No change in status from current to expected -> Not critical
        else:
            return 100
        
    reduction_avail_bikes = (len(battery_levels_after)/len(battery_levels_current))
    average_battery_after = (sum(battery_levels_after)/len(battery_levels_after))

    return reduction_avail_bikes * average_battery_after

def calculate_station_type(target_state, exp_num):
    """
    Returns if station is a pick up, delivery or balanced
    Parameters:
    - target_state = ideal number of bikes at a station
    - LOCATION_TYPE_MARGIN = percentage threshold to categorize as unbalanced
    """
    deviation_from_target = exp_num - target_state

    # If difference from target is higher than percentage of target state it is categorized as unbalanced
    if deviation_from_target > LOCATION_TYPE_MARGIN * target_state:
        station_type = 'p'
    elif deviation_from_target < - LOCATION_TYPE_MARGIN *target_state:
        station_type = 'd'
    else:
        station_type = 'b'
    
    return station_type

def normalize_results(criticalities, time_to_violation_list, deviation_list, neighborhood_crit_list, demand_crit_list, driving_time_list, BL_composition_list):
    """
    Normalizes all the scores so they go from 0 to 1, where 1 is the most critical possible for each category.

    Parameters:
    - criticalities = dictionary, key = potential station, value = list of criticality scores
    - time_to_violation_list = list of all scores from potential stations
    - deviation_list = list of all scores from potential stations
    - neighborhood_crit_list = list of all scores from potential stations
    - demand_crit_list = list of all scores from potential stations
    - driving_time_list = list of all scores from potential stations
    - BL_composition_list = list of all scores from potential stations
    """
    # Find min and max of all categories, used to normalize later
    max_time = max(time_to_violation_list)
    min_time = min(time_to_violation_list)

    max_deviation = max(deviation_list)
    min_deviation = min(deviation_list)

    max_neighborhood = max(neighborhood_crit_list)
    min_neighborhood = min(neighborhood_crit_list)

    max_demand = max(demand_crit_list)
    min_demand = min(demand_crit_list)

    max_driving_time = max(driving_time_list)
    min_driving_time = min(driving_time_list)

    max_battery_level = max(BL_composition_list)
    min_battery_level = min(BL_composition_list)

    criticalities_normalized = dict()
    for station in criticalities:
        # Find the normalized score from scores made
        normalized_violation_crit = find_relative_score(criticalities[station][0], max_time, min_time)
        normalized_deviation_crit = find_relative_score(criticalities[station][1], max_deviation, min_deviation)
        normalized_neighborhood_crit = find_relative_score(criticalities[station][2], max_neighborhood, min_neighborhood)
        normalized_demand_crit = find_relative_score(criticalities[station][3], max_demand, min_demand)
        normalize_driving = find_relative_score(criticalities[station][4], max_driving_time, min_driving_time)
        normalize_battery = find_relative_score(criticalities[station][5], max_battery_level, min_battery_level)

        # Add the scores into the returned dictionary, adjusted for so that the higher the more critical it is
        criticalities_normalized[station] = [
            1 - normalized_violation_crit if normalized_violation_crit else 0,
            normalized_deviation_crit if normalized_deviation_crit else 0,
            normalized_neighborhood_crit if normalized_neighborhood_crit else 0,
            normalized_demand_crit if normalized_demand_crit else 0,
            1 - normalize_driving if normalize_driving else 0,
            1 - normalize_battery if normalize_battery else 0
        ]

    return criticalities_normalized


def find_relative_score(score, max, min):
    """
    Returns the score in a normalized matter between 0 and 1 (Min-Max matter).
    If score = min -> return 0
    If score = max -> return 1
    """
    return None if max == min else (score - min)/(max - min)