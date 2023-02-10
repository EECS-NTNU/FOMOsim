from policies.gleditsch_hagen.utils import calculate_time_to_violation

#alternatively take in station object directly?
def calculate_criticality(weights, simul, station_id, max_time_to_violation, max_deviation, max_neighborhood_crit, max_net_demand):
    [w_t, w_dev, w_n, w_dem] = weights
    TIME_PERIOD = 60
    station = simul.state.stations[station_id]
    t_state = station.get_target_state(simul.day(), simul.hour())
    if station.number_of_bikes() - t_state > 0:
        station_type = "p"  #pick-up
    elif station.number_of_bikes() - t_state < 0:
        station_type = "d"  #delivery
    else:
        station_type = "b"  #balanced
    
    time_to_violation = calculate_time_to_violation #check if this works
    time_to_violation_normalized = 0    #incase if=False
    if max_time_to_violation > 0:
        time_to_violation_normalized = time_to_violation/max_time_to_violation

    deviation_from_t_state = abs(t_state - station.number_of_bikes())
    deviation_from_t_state_normalized = 0
    if max_deviation > 0:
        deviation_from_t_state_normalized = deviation_from_t_state/max_deviation

    neighborhood_crit = calculate_neighborhood_criticality(simul, station)
    neighborhood_crit_normalized = 0
    if max_neighborhood_crit > 0:
        neighborhood_crit_normalized = neighborhood_crit/max_neighborhood_crit

    net_demand = (TIME_PERIOD/60)*(station.get_arrive_intensity(simul.day(), simul.hour()) - station.get_leave_intensity(simul.day(), simul.hour()))
    demand_crit = 0
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

    return w_t*time_to_violation_normalized + w_dev*deviation_from_t_state_normalized + w_n*neighborhood_crit_normalized + w_dem*demand_crit_normalized

def calculate_neighborhood_criticality(simul, station):
    # TODO
    return