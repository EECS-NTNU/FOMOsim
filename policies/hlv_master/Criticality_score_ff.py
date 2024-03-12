from .Simple_calculations import calculate_net_demand, calculate_hourly_discharge_rate
from settings import *

"""
Calculates the criticality score for a location to be visited by a vehicle.
Criticality score is based on:
- Time to violation = How long is it until a starvation/congestion appears
- Deviation from target state = The difference from the locations inventory and the current target state
- Neighborhood criticality = How much can neighboring locations affect the current locations criticality
- Demand = How the current demand contributes to the criticality
- Driving time = The time it takes for the vehicle to arrive contributes to the criticality
"""

def calculate_criticality(weights, simul, potential_clusters, cluster, cluster_type, total_num_bikes_in_system, visited_areas = None):

    [w_t, w_dev, w_n, w_dem, w_dri, w_bc] = weights
    TIME_HORIZON = 60
    criticalities = dict() # key: cluster, value: list of values for factors to consider.
    criticalities_summed = dict()

    time_to_violation_list = []
    deviation_list = []
    neighborhood_crit_list = [] 
    demand_crit_list = []
    driving_time_list = [] 
    BL_composition_list = []

    for potential_station in potential_clusters:
        net_demand = calculate_net_demand(potential_station, simul.time, simul.day(), simul.hour(), 60)
        target_state = potential_station.get_target_state(simul.day(),simul.hour())
        expected_num_escooters = potential_station.number_of_bikes() - len(potential_station.get_swappable_bikes(20)) + net_demand

        if cluster_type == 'p':
            potential_station_type = 'p'
        elif cluster_type == 'd':
            potential_station_type = 'd'
        else:
            potential_station_type = calculate_station_type(target_state, expected_num_escooters)

        # Time to violation
        time_to_violation = calculate_time_to_violation(net_demand, potential_station, simul, total_num_bikes_in_system)
        time_to_violation_list.append(time_to_violation)

        # Deviation from target state
        deviation_from_target_state = calculate_deviation_from_target_state(potential_station, net_demand, target_state)
        deviation_list.append(deviation_from_target_state)

        # Neighborhood criticality
        neighborhood_crit = calculate_neighborhood_criticality(simul, potential_station, TIME_HORIZON, cluster_type, visited_areas)
        neighborhood_crit_list.append(neighborhood_crit)

        # Demand criticality
        demand_crit = calculate_demand_criticality(cluster_type, net_demand)
        demand_crit_list.append(demand_crit)

        # Driving time criticality
        driving_time_crit = calculate_driving_time_crit(simul, cluster, potential_station)
        driving_time_list.append(driving_time_crit)

        # Battery level composition criticality
        battery_level_comp_crit = calculate_battery_level_composition_criticality(simul, potential_station, total_num_bikes_in_system)
        BL_composition_list.append(battery_level_comp_crit)

        criticalities[potential_station] = [time_to_violation, deviation_from_target_state, neighborhood_crit, demand_crit, driving_time_crit, battery_level_comp_crit]
    
    criticalities_normalized = normalize_results(criticalities, time_to_violation_list, deviation_list, neighborhood_crit_list, demand_crit_list, driving_time_list, BL_composition_list)

    for cluster in criticalities_normalized:
        criticalities_normalized[cluster][0] *= w_t
        criticalities_normalized[cluster][1] *= w_dev
        criticalities_normalized[cluster][2] *= w_n
        criticalities_normalized[cluster][3] *= w_dem
        criticalities_normalized[cluster][4] *= w_dri
        criticalities_normalized[cluster][5] *= w_bc

        criticalities_summed[cluster] = criticalities_normalized[cluster][0] + criticalities_normalized[cluster][1] 
        + criticalities_normalized[cluster][2] + criticalities_normalized[cluster][3] + criticalities_normalized[cluster][4] + criticalities_normalized[cluster][5]

    #Sorted in descending order - most critical first
    criticalities_summed = dict(sorted(criticalities_summed.items(), key=lambda item: item[1], reverse=True))

    return criticalities_summed
    
def calculate_time_to_violation(net_demand, cluster, simul, total_num_bikes_in_system):
    """
    Returns a float-score of how critical the cluster is based on how long it is until a starvation occurs.
    A low score is critical, while high is less critical

    Parameters:
    - net_demand = float, the net demand calculated from the location's arrive/leave intensities
    - cluster = Cluster-object under consideration
    - simul = Simulator
    - total_num_bikes_in_system = total number of bikes in the system
    """
    time_to_violation = 0
    
    # Calculate time if a starvation might occur
    if net_demand < 0:
        sorted_bikes_by_battery = sorted(cluster.bikes.values(), key=lambda bike: bike.battery, reverse=False)

        # Time until there is no bikes left at an area that has sufficiant battery level
        violation_demand = len(cluster.get_available_bikes()) / -net_demand
        
        # Calculate the time until there are only bikes at area with too low battery for rental, based on hourly discharge in system
        bikes_most_charged = [bike.battery for bike in sorted_bikes_by_battery[-3:]]
        average_battery_top3 = sum(bikes_most_charged)/len(bikes_most_charged)
        battery_over_limit_top3 = average_battery_top3 - BATTERY_LIMIT_TO_USE
        violation_battery = battery_over_limit_top3 / calculate_hourly_discharge_rate(simul, total_num_bikes_in_system, False)

        time_to_violation = min(
            violation_demand, 
            violation_battery
        )

    # Set time to violation far in the future, since no starvation can happen with positive net_demand (no capacity)
    else:
        time_to_violation = 8
    
    # The highest value allowed is 8 hours
    return min(time_to_violation, 8)

def calculate_deviation_from_target_state(cluster, net_demand, target_state):
    """
    Returns int-score for the deviation criticality, number of bikes away from the target_state.
    The higher the score, the more critical. 

    Parameters:
    - cluster = Cluster-object under consideration
    - net_demand = expected demand for the cluster during the time horizon
    - target_state = number of bikes ideally to be at the cluster
    """

    available_bikes_future = len(cluster.get_available_bikes()) + net_demand

    # If there is expected to still be bikes at cluster find the difference
    if available_bikes_future > 0:
        deviation_from_target_state = abs(target_state - available_bikes_future)
    else:
        deviation_from_target_state = target_state
    
    return deviation_from_target_state

def calculate_neighborhood_criticality(simul, station, TIME_HORIZON, station_type, visited_stations):
    neighborhood_crit = 0
    neighbors = station.get_neighbours()
    
    for neighbor in neighbors:
        station_crit = 0
        
     
        neighbor_demand = calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(), 60)
        neighbor_target_state = neighbor.get_target_state(simul.day(),simul.hour())

        expected_number_escooters = (neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(20)) + neighbor_demand)
        neighbor_type = calculate_station_type(neighbor_target_state, expected_number_escooters)

            #Extra critical because of neighbor with same status
        if neighbor_type == station_type:
            station_crit += 1

            #Less critical because of neghbor whit complimentary status
            #Mabye change >0 into > % of target state?
        if station_type == 'p' and neighbor_target_state - expected_number_escooters > neighbor_target_state * NEIGHBOR_BALANCE_PICKUP:
            station_crit -= 1
            
        elif station_type == 'd' and expected_number_escooters - neighbor_target_state > neighbor_target_state * NEIGHBOR_BALANCE_DELIVERY:
            station_crit -= 1
            
        elif station_type == 'b':
            station_crit -= 1
            
            #If demand betters or worsen the situation over time
        if station_type == neighbor_type:
            station_crit += calculate_demand_criticality(neighbor_type, neighbor_demand)
            
            #Battery level composition
        if station_type == 'd' and USE_BATTERY_CRITICALITY:
            current_escooters = neighbor.bikes
            battery_levels_neighbor = [escooter.battery for escooter in current_escooters.values() if escooter.battery > 20]
            avg_battery_level = (sum(battery_levels_neighbor) / len(battery_levels_neighbor)) if len(battery_levels_neighbor) > 0 else 0

            if neighbor_type == 'd':
                if avg_battery_level < 50:
                    station_crit += 1
            

        
        #Accounts for distance, closer is better i think
        """ distance = (simul.state.traveltime_vehicle_matrix[(station.location_id, neighbor.location_id)]/60)*VEHICLE_SPEED
        station_crit *= (1 - (distance/MAX_ROAMING_DISTANCE_SOLUTIONS)) """

        neighborhood_crit += station_crit
    
    return neighborhood_crit

def calculate_demand_criticality(cluster_type, net_demand):
    """
    Returns the criticality score based on demand and station type.
    The higher the score, the more critical.

    Parameters:
    - station_type = character describing if the station under consideration is a pick up, delivery or balanced
    - net_demand = how the flow at the station is for the set time_horizon
    """

    # The cluster has too many bikes and more are arriving -> positive
    if cluster_type == "p" and net_demand > 0:
        demand_crit = net_demand
    
    # The cluster has too many bikes and bikes are departuring -> positive
    elif cluster_type == "p" and net_demand < 0:
        demand_crit = -net_demand
    
    # The cluster had too few bikes but bikes are arriving here -> negative
    elif cluster_type == "d" and net_demand > 0:
        demand_crit = -net_demand

    # The cluster had too few bikes, and bikes are departuring -> negative
    elif cluster_type == "d" and net_demand < 0:
        demand_crit = net_demand
    
    # Cluster is at target state
    elif cluster_type == "b":
        demand_crit = 0 #TODO gir dette mening?
    
    # net_demand == 0
    else:
        demand_crit = 0
    
    return demand_crit

def calculate_driving_time_crit(simul, current_cluster, potential_cluster):
    """
    Returns the driving time between two cluster centers.
    The lower the score the more critical.
    """
    return simul.state.get_vehicle_travel_time(current_cluster.location_id, potential_cluster.location_id)

def calculate_battery_level_composition_criticality(simul, station, total_num_bikes_in_system):
    """
    Returns a float-score based on how the battery levels may be in the future.
    The lower the score, the more critical.

    Parameters:
    - simul = Simulator
    - station = Station-object under consideration
    - total_num_bikes_in_system = total number of bikes in the system
    """
    current_escooters = station.bikes
    hourly_discharge_rate = calculate_hourly_discharge_rate(simul, total_num_bikes_in_system, False)

    # Make list of the batteries of usable bikes at the station in current time and the next hour 
    battery_levels_current = [escooter.battery for escooter in current_escooters.values() if escooter.usable()]
    battery_levels_after = [escooter.battery - hourly_discharge_rate for escooter in current_escooters.values() if (escooter.battery - hourly_discharge_rate) > BATTERY_LIMIT_TO_USE]

    # Very critical if there are no bikes with sufficient battery left
    if len(battery_levels_after) == 0 or len(battery_levels_current) == 0:
        if len(battery_levels_after) < len(battery_levels_current):
            return 0
        else:
            return 10 # TODO hvilket tall er ikke kritisk?
        
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
    allowed_deviation = LOCATION_TYPE_MARGIN * target_state

    # If difference from target is higher than percentage of target state it is categorized as unbalanced
    if deviation_from_target > allowed_deviation:
        station_type = 'p'
    elif deviation_from_target < - allowed_deviation:
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
            1 - normalized_violation_crit if normalized_violation_crit is not None else 0,
            normalized_deviation_crit if normalized_deviation_crit is not None else 0,
            normalized_neighborhood_crit if normalized_neighborhood_crit is not None else 0,
            normalized_demand_crit if normalized_demand_crit is not None else 0,
            1 - normalize_driving if normalize_driving is not None else 0,
            1 - normalize_battery if normalize_battery is not None else 0
        ]
    return criticalities_normalized

def find_relative_score(score, max, min):
    """
    Returns the score in a normalized matter between 0 and 1 (Min-Max matter).
    If score = min -> return 0
    If score = max -> return 1
    """
    return None if max == min else (score - min)/(max - min)