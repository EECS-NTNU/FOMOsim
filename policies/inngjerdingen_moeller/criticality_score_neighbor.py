from policies.gleditsch_hagen.utils import calculate_time_to_violation
from settings import MAX_ROAMING_DISTANCE, BIKE_SPEED

def calculate_criticality(weights, simul, current_station, potential_stations, max_time_to_violation, max_deviation, max_neighborhood_crit, max_net_demand):
    # COMMON DATA
    [w_t, w_dev, w_n, w_dem] = weights
    TIME_HORIZON = 60
    criticalities = dict()      #{station:criticality}

    # CALCULATE CRITICALITY FOR EACH POTENTIAL STATION
    for potential_station in potential_stations:
        net_demand = (TIME_HORIZON/60)*(potential_station.get_arrive_intensity(simul.day(), simul.hour()) - potential_station.get_leave_intensity(simul.day(), simul.hour()))
        t_state = potential_station.get_target_state(simul.day(), simul.hour())
        station_type, exp_num_bikes = calculate_station_type(potential_station, net_demand, t_state)
        
        time_to_violation = calculate_time_to_violation(net_demand, potential_station) #check if this works (hours)
        time_to_violation_normalized = 0    #incase if=False
        if max_time_to_violation > 0:
            time_to_violation_normalized = 1-(time_to_violation/max_time_to_violation)

        deviation_from_t_state = abs(t_state - potential_station.number_of_bikes())
        deviation_from_t_state_normalized = 0
        if max_deviation > 0:
            deviation_from_t_state_normalized = deviation_from_t_state/max_deviation

        neighborhood_crit = calculate_neighborhood_criticality(simul, potential_station, TIME_HORIZON, station_type, max_net_demand)
        neighborhood_crit_normalized = 0
        if max_neighborhood_crit > 0:
            neighborhood_crit_normalized = neighborhood_crit/max_neighborhood_crit

        demand_crit_normalized = calculate_demand_criticality(station_type, net_demand, max_net_demand)
        criticalities[potential_station] = w_t*time_to_violation_normalized + w_dev*deviation_from_t_state_normalized + w_n*neighborhood_crit_normalized + w_dem*demand_crit_normalized


    #sort the dict?
    return criticalities


def calculate_neighborhood_criticality(simul, potential_station, TIME_HORIZON, station_type, max_net_demand):
    neighborhood_crit = 0
    neighbors = potential_station.neighboring_stations  #list
    for neighbor in neighbors:
        station_crit = 0
        neighbor_demand = (TIME_HORIZON/60)*(potential_station.get_arrive_intensity(simul.day(), simul.hour()) - potential_station.get_leave_intensity(simul.day(), simul.hour()))
        neighbor_t_state = neighbor.get_target_state(simul.day(), simul.hour())
        # Similarly imbalanced (+)
        neighbor_type, exp_num_bikes = calculate_station_type(neighbor, neighbor_demand, neighbor_t_state)
        if neighbor_type == station_type:
            station_crit += 1
        # Absorb demand (-)
        if station_type == 'p' and neighbor.capacity - exp_num_bikes > 0:
            station_crit -= 1
        elif station_type == 'd' and exp_num_bikes > 0:
            station_crit -= 1
        # Neighbor demand (higher+)
        station_crit += calculate_demand_criticality(neighbor_type, neighbor_demand, max_net_demand)
        # Distance scaling (closer+, further-)
        distance = simul.state.traveltime_matrix[potential_station.id][neighbor.id]*BIKE_SPEED
        station_crit = station_crit*(1-(distance/MAX_ROAMING_DISTANCE))

        neighborhood_crit += station_crit

    return neighborhood_crit

def calculate_station_type(potential_station, net_demand, t_state):
    exp_num_bikes = potential_station.number_of_bikes() + net_demand
    if exp_num_bikes - t_state > 0:
        station_type = "p"  #pick-up
    elif exp_num_bikes - t_state < 0:
        station_type = "d"  #delivery
    else:
        station_type = "b"  #balanced
    return station_type, exp_num_bikes


def calculate_demand_criticality(station_type, net_demand, max_net_demand):
    #check if demand improves or worsens balance
    if station_type == "p" and net_demand > 0:
        demand_crit = net_demand
    elif station_type == "p" and net_demand < 0:
        demand_crit = -net_demand
    elif station_type == "d" and net_demand > 0:
        demand_crit = -net_demand
    elif station_type == "d" and net_demand < 0:
        demand_crit = net_demand
    elif station_type == "b":
        demand_crit = abs(net_demand)
    if max_net_demand > 0:
        demand_crit_normalized = demand_crit/max_net_demand
    return demand_crit_normalized