from policies.hlv_master.Simple_calculations import calculate_net_demand
from policies.hlv_master.FF_Criticality_score import calculate_time_to_violation as calculate_time_to_violation_FF, calculate_deviation_from_target_state as calculate_deviation_from_target_state_FF, calculate_neighborhood_criticality as calculate_neighborhood_criticality_FF, calculate_demand_criticality as calculate_demand_criticality_FF, calculate_driving_time_crit as calculate_driving_time_crit_FF, calculate_battery_level_composition_criticality as calculate_battery_level_composition_criticality_FF, calculate_cluster_type, normalize_results as normalize_results_FF
from policies.hlv_master.SB_Criticality_score import calculate_time_to_violation as calculate_time_to_violation_SB, calculate_deviation_from_target_state as calculate_deviation_from_target_state_SB, calculate_neighborhood_criticality as calculate_neighborhood_criticality_SB, calculate_demand_criticality as calculate_demand_criticality_SB, calculate_driving_time_crit as calculate_driving_time_crit_SB, calculate_battery_level_composition_criticality as calculate_battery_level_composition_criticality_SB, calculate_station_type, normalize_results as normalize_results_SB

from settings import *
import sim
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

def calculate_criticality_ff(weights, simul, potential_clusters, station, total_num_escooters_in_system, visited_clusters = None):
    """
    Returns a dictionary with station and criticality score for all potential station to visit. The higher the score the more critical the station is.

    Parameters:
    - weights = list of the weights of how much each component matters. Float from 0-1, summed to 1
    - simul = Simulator
    - potential_clusters= list of Cluster-objects to consider visiting
    - total_num_bikes_in_system = int, number of bikes total
    - visited_cluster = tabu_list og clusters that should not be visited
    """
    [w_t, w_dev, w_n, w_dem, w_dri, w_bc] = weights
    criticalities = dict() # key: cluster, value: list of criticality scores for each category
    criticalities_summed = dict() # key: cluster, value: criticality of station

    # Lists of criticality scores for all potential_stations
    time_to_violation_list = []
    deviation_list = []
    neighborhood_crit_list = [] 
    demand_crit_list = []
    driving_time_list = [] 
    BL_composition_list = []

    for potential_cluster in potential_clusters:
        net_demand = calculate_net_demand(potential_cluster, simul.time, simul.day(), simul.hour(), TIME_HORIZON)
        target_state = potential_cluster.get_target_state(simul.day(),simul.hour())
        potential_cluster_type = calculate_cluster_type(target_state, len(potential_cluster.get_available_bikes()) + net_demand)

        # Time to violation
        time_to_violation = calculate_time_to_violation_FF(net_demand, potential_cluster, simul, total_num_escooters_in_system)
        time_to_violation_list.append(time_to_violation)

        # Deviation from target state
        deviation_from_target_state = calculate_deviation_from_target_state_FF(potential_cluster, net_demand, target_state)
        deviation_list.append(deviation_from_target_state)

        # Neighborhood criticality
        neighborhood_crit = calculate_neighborhood_criticality_FF(simul, potential_cluster, potential_cluster_type)
        neighborhood_crit_list.append(neighborhood_crit)

        # Demand criticality
        demand_crit = calculate_demand_criticality_FF(potential_cluster_type, net_demand)
        demand_crit_list.append(demand_crit)

        # Driving time criticality
        driving_time_crit = calculate_driving_time_crit_FF(simul, station, potential_cluster)
        driving_time_list.append(driving_time_crit)

        # Battery level composition criticality
        battery_level_comp_crit = calculate_battery_level_composition_criticality_FF(simul, potential_cluster, total_num_escooters_in_system)
        BL_composition_list.append(battery_level_comp_crit)

        criticalities[potential_cluster] = [time_to_violation, deviation_from_target_state, neighborhood_crit, demand_crit, driving_time_crit, battery_level_comp_crit]
    
    criticalities_normalized = normalize_results_FF(criticalities, time_to_violation_list, deviation_list, neighborhood_crit_list, demand_crit_list, driving_time_list, BL_composition_list)

    for station in criticalities_normalized:
        criticalities_normalized[station][0] *= w_t
        criticalities_normalized[station][1] *= w_dev
        criticalities_normalized[station][2] *= w_n
        criticalities_normalized[station][3] *= w_dem
        criticalities_normalized[station][4] *= w_dri
        criticalities_normalized[station][5] *= w_bc

        # TODO gir dette mening?
        criticalities_normalized[station][0] += ADJUSTING_CRITICALITY
        criticalities_normalized[station][1] += ADJUSTING_CRITICALITY
        criticalities_normalized[station][2] += ADJUSTING_CRITICALITY
        criticalities_normalized[station][3] += ADJUSTING_CRITICALITY
        criticalities_normalized[station][4] += ADJUSTING_CRITICALITY
        criticalities_normalized[station][5] += ADJUSTING_CRITICALITY

        criticalities_summed[station] = criticalities_normalized[station][0] + criticalities_normalized[station][1] 
        + criticalities_normalized[station][2] + criticalities_normalized[station][3] + criticalities_normalized[station][4] + criticalities_normalized[station][5]

    #Sorted in descending order - most critical first
    criticalities_summed = dict(sorted(criticalities_summed.items(), key=lambda item: item[1], reverse=True))

    return criticalities_summed

def calculate_criticality_sb(weights, simul, potential_stations, station, total_num_bikes_in_system, visited_stations = None):
    """
    Returns a dictionary with station and criticality score for all potential station to visit. The higher the score the more critical the station is.

    Parameters:
    - weights = list of the weights of how much each component matters. Float from 0-1, summed to 1
    - simul = Simulator
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
        net_demand = calculate_net_demand(potential_station, simul.time, simul.day(), simul.hour(), TIME_HORIZON)
        target_state = potential_station.get_target_state(simul.day(),simul.hour())
        potential_station_type = calculate_station_type(target_state, len(potential_station.get_available_bikes()) + net_demand)

        # Time to violation
        time_to_violation = calculate_time_to_violation_SB(net_demand, potential_station, simul, total_num_bikes_in_system)
        time_to_violation_list.append(time_to_violation)

        # Deviation from target state
        deviation_from_target_state = calculate_deviation_from_target_state_SB(potential_station, net_demand, target_state)
        deviation_list.append(deviation_from_target_state)

        # Neighborhood criticality
        neighborhood_crit = calculate_neighborhood_criticality_SB(simul, potential_station, potential_station_type, visited_stations)
        neighborhood_crit_list.append(neighborhood_crit)

        # Demand criticality
        demand_crit = calculate_demand_criticality_SB(potential_station_type, net_demand)
        demand_crit_list.append(demand_crit)

        # Driving time criticality
        driving_time_crit = calculate_driving_time_crit_SB(simul, station, potential_station)
        driving_time_list.append(driving_time_crit)

        # Battery level composition criticality
        battery_level_comp_crit = calculate_battery_level_composition_criticality_SB(simul, potential_station, total_num_bikes_in_system)
        BL_composition_list.append(battery_level_comp_crit)

        criticalities[potential_station] = [time_to_violation, deviation_from_target_state, neighborhood_crit, demand_crit, driving_time_crit, battery_level_comp_crit]
    
    criticalities_normalized = normalize_results_SB(criticalities, time_to_violation_list, deviation_list, neighborhood_crit_list, demand_crit_list, driving_time_list, BL_composition_list)

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