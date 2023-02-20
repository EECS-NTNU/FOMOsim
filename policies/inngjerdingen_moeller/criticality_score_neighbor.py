from policies.gleditsch_hagen.utils import calculate_time_to_violation
from settings import MAX_ROAMING_DISTANCE, BIKE_SPEED

def calculate_criticality(weights, simul, potential_stations):
    # COMMON DATA
    [w_t, w_dev, w_n, w_dem] = weights
    TIME_HORIZON = 60
    criticalities = dict()      #{station:[time_to_viol, dev_t_state, neigh_crit, dem_crit]}
    criticalities_summed = dict()   #{station:crit_sum}
    time_to_violation_list = []
    deviation_list = []
    neighborhood_crit_list = []
    demand_crit_list = []


    # CALCULATE CRITICALITY FOR EACH POTENTIAL STATION
    for potential_station in potential_stations:
        net_demand = (TIME_HORIZON/60)*(potential_station.get_arrive_intensity(simul.day(), simul.hour()) - potential_station.get_leave_intensity(simul.day(), simul.hour()))
        t_state = potential_station.get_target_state(simul.day(), simul.hour())
        station_type, exp_num_bikes = calculate_station_type(potential_station, net_demand, t_state)
        
        time_to_violation = calculate_time_to_violation(net_demand, potential_station) #check if this works (hours) and check numbers returned
        time_to_violation_list.append(time_to_violation)

        deviation_from_t_state = abs(t_state - potential_station.number_of_bikes() - net_demand) # alt. include exp. demand during TH
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
        criticalities_summed[station] = criticalities_normalized[station][0] + criticalities_normalized[station][1] 
        + criticalities_normalized[station][2] + criticalities_normalized[station][3]
    

    #sort the dict?
    return criticalities_summed


def calculate_neighborhood_criticality(simul, potential_station, TIME_HORIZON, station_type):
    neighborhood_crit = 0
    neighbors = potential_station.neighboring_stations  #list

    for neighbor in neighbors:
        station_crit = 0
        # neighbor_demand = (TIME_HORIZON/60)*(potential_station.get_arrive_intensity(simul.day(), simul.hour()) - potential_station.get_leave_intensity(simul.day(), simul.hour())) #SM: is this supposed to be neighbor.get_arrive_intensity etc. 
        neighbor_demand = (TIME_HORIZON/60)*(neighbor.get_arrive_intensity(simul.day(), simul.hour()) - neighbor.get_leave_intensity(simul.day(), simul.hour()))
        neighbor_t_state = neighbor.get_target_state(simul.day(), simul.hour())
        
        # Similarly imbalanced (+)
        neighbor_type, exp_num_bikes = calculate_station_type(neighbor, neighbor_demand, neighbor_t_state) #SM: can we avoid doing these calculations multiple times for each station? 
        if neighbor_type == station_type:
            station_crit += 1
        
        # Absorb demand (-)
        if station_type == 'p' and neighbor.capacity - exp_num_bikes > 0: 
            station_crit -= 1
        elif station_type == 'd' and exp_num_bikes > 0:
            station_crit -= 1

        
        # Neighbor demand (higher+)
        if station_type == neighbor_type:
            station_crit += calculate_demand_criticality(neighbor_type, neighbor_demand)    #OBS: not normalized
        
        # Distance scaling (closer+, further-)
        distance = simul.state.traveltime_matrix[potential_station.id][neighbor.id]*BIKE_SPEED
        station_crit *= (1-(distance/MAX_ROAMING_DISTANCE))

        neighborhood_crit += station_crit

    return neighborhood_crit

def calculate_station_type(potential_station, net_demand, t_state):
    margin = 0.15    # slack around t_state 15%
    exp_num_bikes = potential_station.number_of_bikes() + net_demand
    if exp_num_bikes - t_state > margin*(potential_station.capacity-t_state):
        station_type = "p"  #pick-up
    elif exp_num_bikes - t_state < -margin*t_state:
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
    else:
        demand_crit = abs(net_demand)
    return demand_crit


def normalize_results(criticalities, time_to_violation_list, deviation_list, neighborhood_crit_list, demand_crit_list):
    max_time = max(max(time_to_violation_list),1)
    max_deviation = max(max(deviation_list),1)
    max_neighborhood = max(max(neighborhood_crit_list),1)
    min_neighborhood = min(min(neighborhood_crit_list),-1)
    max_demand = max(max(demand_crit_list),1)  #this can potentially be 0, happend for 1 out of 10 seeds
    criticalities_normalized = dict()
    for station in criticalities:
        criticalities_normalized[station] = []
        criticalities_normalized[station].append(1-criticalities[station][0]/max_time)
        criticalities_normalized[station].append(criticalities[station][1]/max_deviation)
        if criticalities[station][2] >= 0:
            criticalities_normalized[station].append(criticalities[station][2]/max_neighborhood)
        else: 
            criticalities_normalized[station].append(criticalities[station][2]/-min_neighborhood)
        criticalities_normalized[station].append(criticalities[station][3]/max_demand)
    return criticalities_normalized
