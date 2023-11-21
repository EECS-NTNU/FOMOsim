from Simple_calculations import calculate_net_demand, calculate_hourly_discharge_rate
from settings import MAX_ROAMING_DISTANCE_SOLUTIONS, VEHICLE_SPEED
from Variables import *

#########################################################
# THIS FILE CONTAINS ALL CRITICALITY SCORE CALCULATIONS #
#########################################################


########################################################################################
# This is where the criticality score is calculated
# Time to voilation, deviations, neighborhood criticality, demand and driving time
# We need to descide how and if we want to calculate in battery levels in this scoore
# If we want to adjust this score this is where we do it
########################################################################################



def calculate_criticality(weights, simul, potential_stations, station, station_type, visited_stations = None):

    #This is where we have to add one weight for battery level if nessesary

    # TODO - add weight for battery_criticality

    [w_t, w_dev, w_n, w_dem, w_dri] = weights
    TIME_HORIZON = 60
    criticalities = dict() # key: station, value: list of values for factors to consider.
    criticalities_summed = dict()

    #The facors in consideration
    time_to_violation_list = []
    deviation_list = []
    neighborhood_crit_list = [] 
    demand_crit_list = []
    driving_time_list = [] 
    BL_composition_list = []

    #Calculate criticality scoore for each potential station
    for potential_station in potential_stations:
        net_demand = calculate_net_demand(potential_station, simul.time, simul.day(), simul.hour(), 60)
        target_state = potential_station.get_target_state(simul.day(),simul.hour())
        expected_num_escooters = potential_station.number_of_bikes() - len(potential_station.get_swappable_bikes(20)) + net_demand

        if station_type == 'p':
            potential_station_type = 'p'
        elif station_type == 'd':
            potential_station_type = 'd'
        else:
            potential_station_type = calculate_station_type(target_state, expected_num_escooters)

        # Time to violation
        time_to_violation = calculate_time_to_violation_IM(net_demand, potential_station, simul)
        time_to_violation_list.append(time_to_violation)

        # Deviation from target state
        deviation_from_target_state = calculate_deviation_from_target_state(potential_station, net_demand, target_state)
        deviation_list.append(deviation_from_target_state)

        # Neighborhood criticality
        neighborhood_crit = calculate_neighborhood_criticality(simul, potential_station, TIME_HORIZON, station_type, visited_stations)
        neighborhood_crit_list.append(neighborhood_crit)

        # Demand criticality
        demand_crit = calculate_demand_criticality(station_type, net_demand)
        demand_crit_list.append(demand_crit)

        # Driving time criticality
        driving_time_crit = calculate_driving_time_crit(simul, station, potential_station)
        driving_time_list.append(driving_time_crit)

        # Battery level composition criticality
        battery_level_comp_crit = calculate_battery_level_composition_criticality(simul, station)
        BL_composition_list.append(battery_level_comp_crit)


        criticalities[potential_station] = [time_to_violation, deviation_from_target_state, neighborhood_crit, demand_crit, driving_time_crit]
    
    criticalities_normalized = normalize_results(criticalities, time_to_violation_list, deviation_list, neighborhood_crit_list, demand_crit_list, driving_time_list)

    for station in criticalities_normalized:
        criticalities_normalized[station][0] *= w_t
        criticalities_normalized[station][1] *= w_dev
        criticalities_normalized[station][2] *= w_n
        criticalities_normalized[station][3] *= w_dem
        criticalities_normalized[station][4] *= w_dri

        criticalities_summed[station] = criticalities_normalized[station][0] + criticalities_normalized[station][1] 
        + criticalities_normalized[station][2] + criticalities_normalized[station][3] + criticalities_normalized[station][4]

    #Sorted in descending order - most critical first
    criticalities_summed = dict(sorted(criticalities_summed.items(), key=lambda item: item[1], reverse=True))

    return criticalities_summed
    



#################################################################################################
# TIME TO VIOLATION calculated here                                                             #
#                                                                                               #
# This is where i suggust that we account for battery level                                     #
# Uses the demand caluclated for the next hour for the forseable future - room for improvement? #
#################################################################################################

def calculate_time_to_violation_IM(net_demand, station,simul):
    time_to_violation = 0

    #since we operate without a station_capacity positive net demand wont lead to violation, should we punich in a way
    if net_demand > 0:
        time_to_violation = 8 
    
    #Starvations will lead to violations -> when demand occurs and the cluster has noe escooters
    #This is the perfect spot to add some calculation on battery degeneration over itme
    elif net_demand < 0:
        sorted_escooters_in_station = sorted(station.bikes.values(), key=lambda bike: bike.battery, reverse=False)
        time_to_violation = min((station.number_of_bikes() - len(station.get_swappable_bikes(20)))/ -net_demand, (sum(Ebike.battery for Ebike in sorted_escooters_in_station[-3:])/3)/(calculate_hourly_discharge_rate(simul)*60))

        #We treat violations >= 8 as the same
        if time_to_violation > 8:
            time_to_violation = 8
    
    else:
        time_to_violation = 8
    
    return time_to_violation



###############################################
# DEVIATION FROM TARGET STATE calculated here #
#                                             #
# Accounted for battery levels                #
############################################### 

def calculate_deviation_from_target_state(station, net_demand, target_state):
    
    num_escooters = (station.number_of_bikes() - station.get_swappable_bikes(20) + net_demand)

    if net_demand > 0:
        deviation_from_target_state = abs(target_state - num_escooters)

    else:
        if num_escooters >= 0:
            deviation_from_target_state = abs(target_state - num_escooters)
        else:
            deviation_from_target_state = target_state
    
    return deviation_from_target_state


###############################################
# NEIGHBORHOOD CRITICALITY calculated here    #
#                                             #
# Accounted for battery levels                #
############################################### 

def calculate_neighborhood_criticality(simul, station, TIME_HORIZON, station_type, visited_stations):
    neighborhood_crit = 0
    neighbors = station.neighboring_stations
    
    for neighbor in neighbors:
        station_crit = 0
        if visited_stations != None and neighbor.id in visited_stations:
            station_crit -= 3
        else:
            neighbor_demand = calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(), 60)
            neighbor_target_state = neighbor.get_target_state(simul.day(),simul.hour())

            expected_number_escooters = (neighbor.number_of_bikes() - neighbor.get_swappable_bikes(20) + neighbor_demand)
            neighbor_type = calculate_station_type(neighbor_target_state, expected_number_escooters)

            #Extra critical because of neighbor with same status
            if neighbor_type == station_type:
                station_crit += 1

            #Less critical because of neghbor whit complimentary status
            #Mabye change >0 into > % of target state?
            if station_type == 'p' and neighbor_target_state - expected_number_escooters > 0:
                station_crit -= 1
            
            elif station_type == 'd' and expected_number_escooters - neighbor_target_state > 0:
                station_crit -= 1
            
            elif station_type == 'b':
                station_crit -= 1
            
            #If demand betters or worsen the situation over time
            if station_type == neighbor_type:
                station_crit += calculate_demand_criticality(neighbor_type, neighbor_demand)
        
        #Accounts for distance, closer is better i think
        distance = (simul.state.traveltime_vehicle_matrix[station.id][neighbor.id]/60)*VEHICLE_SPEED
        station_crit *= (1 - (distance/MAX_ROAMING_DISTANCE_SOLUTIONS))

        neighborhood_crit += station_crit
    
    return neighborhood_crit



##########################################################################
# DEMAND CRITICALITY calculated here                                     #
#                                                                        #
# If the natural flow betters the situation the station is less critical #                                             
########################################################################## 


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



##########################################################################
# DRIVING TIME CRITICALITY calculated here                               #
#                                                                        #          
##########################################################################


def calculate_driving_time_crit(simul, current_station, potential_station):
    return simul.state.get_vehicle_travel_time(current_station.id, potential_station.id)



#########################################################
# BATTERY LEVEL COMPOSITION CRITICALITY calculated here #
# NEW criticality factor                                #
#########################################################

def calculate_battery_level_composition_criticality(simul, station):
    
     current_escooters = station.bikes
     hourly_discharge_rate = calculate_hourly_discharge_rate(simul) * 60

     battery_levels_current = [escooter.battry for escooter in current_escooters if escooter.battery > 20]
     battery_levels_after = [escooter.battery - hourly_discharge_rate for escooter in current_escooters if (escooter.battery - hourly_discharge_rate) > 20]

     #Apply weighted avarage functionality here if we want

     return (len(battery_levels_after)/ len(battery_levels_current))*(sum(battery_levels_after)/len(battery_levels_after))






#Simple calculation functions to help in the functions above 

def calculate_station_type(target_state, exp_num):
    margin = 0.15 #set in settings sheeme

    if exp_num - target_state > margin*target_state:
        station_type = 'p'
    elif exp_num - target_state < - margin*target_state:
        station_type = 'd'
    else:
        station_type = 'b'
    
    return station_type


def normalize_results(criticalities, time_to_violation_list, deviation_list, neighborhood_crit_list, demand_crit_list, driving_time_list):
    max_time = max(max(time_to_violation_list),1)
    max_deviation = max(max(deviation_list),1)
    max_neighborhood = max(max(neighborhood_crit_list),1)
    min_neighborhood = min(min(neighborhood_crit_list),-1)
    max_demand = max(max(demand_crit_list),1)  #this can potentially be 0, happend for 1 out of 10 seeds
    max_driving_time = max(driving_time_list)
    
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
        criticalities_normalized[station].append(1-criticalities[station][4]/max_driving_time)
    return criticalities_normalized



