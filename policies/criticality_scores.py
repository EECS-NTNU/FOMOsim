from policies.gleditsch_hagen.utils import calculate_net_demand, calculate_time_to_violation


def calculate_criticality_normalized(weights,simul,start_station_id, potential_station_id,net_demand_ps,
    max_net_demand=1,max_driving_time=1,max_time_to_violation=1,max_deviation=1, #default is NO normalization
    planning_horizon=60): #not used 
 
    [omega1,omega2,omega3,omega4] = weights

    potential_station = simul.state.stations[potential_station_id]  
    
    driving_time = simul.state.get_vehicle_travel_time(start_station_id,potential_station_id)
    driving_time_normalized = driving_time /max_driving_time #function of max, mean, median,...?
    #OR USE max(simul.state.traveltime_matrix)
    
    #net demand for LOCKS (so positive values is parking of bikes)
    #THIS SHOULD HAVE BEEN ABSOLUTE VALUE OF NET DEMAND. THIS IS NOT CLEAR IN THE PAPER!!!
    net_demand_absolute = abs(net_demand_ps)
    net_demand_normalized = 0
    if max_net_demand != 0:
        net_demand_normalized = net_demand_absolute/max_net_demand
    
    #ALTERNATIVELY: use station.get_arrive_intensity(day,hour) and station.get_leave_intensity(day,hour)
    time_to_violation = calculate_time_to_violation(net_demand_ps,potential_station)
    time_to_violation_normalized = 1
    if max_time_to_violation > 0:
        time_to_violation_normalized = time_to_violation/max_time_to_violation
    
    target_state = potential_station.get_target_state(simul.day(), simul.hour())
    deviation_from_target_state = abs(target_state-len(potential_station.bikes))
    deviation_from_target_state_normalized = 0
    if max_deviation > 0:
        deviation_from_target_state_normalized = deviation_from_target_state/max_deviation
    
    return round((
        - omega1*time_to_violation_normalized  
        + omega2*net_demand_normalized
        - omega3*driving_time_normalized  
        + omega4*deviation_from_target_state_normalized
    ),3)
    #high value implies a critical station




if __name__ == "__main__":

    #some testing
   weights = [1,2]
   [w1,w2] = weights
   print(w1) 