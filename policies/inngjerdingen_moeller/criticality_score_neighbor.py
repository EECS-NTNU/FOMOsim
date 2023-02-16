from policies.gleditsch_hagen.utils import calculate_time_to_violation
from settings import MAX_ROAMING_DISTANCE, BIKE_SPEED

def calculate_criticality(weights, simul, current_station, potential_stations):
    # COMMON DATA
    [w_t, w_dev, w_n, w_dem] = weights
    TIME_HORIZON = 60
    criticalities = dict()      #{station:[criticalities]}
    criticalities_summed = dict()
    time_to_violation_list = []
    deviation_list = []
    neighborhood_crit_list = []
    demand_crit_list = []


    # CALCULATE CRITICALITY FOR EACH POTENTIAL STATION
    for potential_station in potential_stations:
        net_demand = (TIME_HORIZON/60)*(potential_station.get_arrive_intensity(simul.day(), simul.hour()) - potential_station.get_leave_intensity(simul.day(), simul.hour()))
        t_state = potential_station.get_target_state(simul.day(), simul.hour())
        station_type, exp_num_bikes = calculate_station_type(potential_station, net_demand, t_state)
        
        time_to_violation = calculate_time_to_violation(net_demand, potential_station) #check if this works (hours)
        time_to_violation_list.append(time_to_violation)

        deviation_from_t_state = abs(t_state - potential_station.number_of_bikes())
        deviation_list.append(deviation_from_t_state)

        neighborhood_crit = calculate_neighborhood_criticality(simul, potential_station, TIME_HORIZON, station_type)
        neighborhood_crit_list.append(neighborhood_crit)

        demand_crit = calculate_demand_criticality(station_type, net_demand)
        demand_crit_list.append(demand_crit)

        criticalities[potential_station] = [time_to_violation, deviation_from_t_state, neighborhood_crit, demand_crit]

    criticalities_normalized = normalize_results(criticalities, time_to_violation_list, deviation_list, neighborhood_crit_list, demand_crit_list)

    # Applying weights
    for station in criticalities_normalized:
        criticalities_normalized[station][0] *= w_t
        criticalities_normalized[station][1] *= w_dev
        criticalities_normalized[station][2] *= w_n
        criticalities_normalized[station][3] *= w_dem
        # Summing all criticalities
        criticalities_summed[station] = criticalities_normalized[station][0] + criticalities_normalized[station][1] +criticalities_normalized[station][2] + criticalities_normalized[station][3]
    

    #sort the dict?
    return criticalities_summed


def calculate_neighborhood_criticality(simul, potential_station, TIME_HORIZON, station_type):
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
        station_crit += calculate_demand_criticality(neighbor_type, neighbor_demand)
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


def calculate_demand_criticality(station_type, net_demand):
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
    return demand_crit


def normalize_results(criticalities, time_to_violation_list, deviation_list, neighborhood_crit_list, demand_crit_list):
    max_time = max(time_to_violation_list)
    max_deviation = max(deviation_list)
    max_neighborhood = max(neighborhood_crit_list)
    max_demand = max(demand_crit_list)
    # CHECK THAT NONE OF THESE ARE = 0
    criticalities_normalized = dict()
    for station in criticalities:
        criticalities_normalized[station] = []
        criticalities_normalized[station].append(1-criticalities[station][0]/max_time)
        criticalities_normalized[station].append(criticalities[station][1]/max_deviation)
        criticalities_normalized[station].append(criticalities[station][2]/max_neighborhood)
        criticalities_normalized[station].append(criticalities[station][3]/max_demand)
    return criticalities_normalized