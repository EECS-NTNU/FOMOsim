# from .Variables import AVERAGE_LENGHT_OF_TRIP
from settings import *

def calculate_net_demand(station, time_now, day, hour, planning_horizon):
    """
    Returns net demand (int) for the given planning horizon. Calculated by the current demand and the demand for the next hour (if needed).

    Parameters:
    - station = Station-object to calculate demand for
    - time_now = current time of the simulator
    - day, hour = day and hour to get the demand at station
    - planning_horizon = time length to check demand for
    """
    if planning_horizon > 60:
        print('Do not support such long time horizen yet')
    
    # Calculate the number of minutes within the current hour has passed
    minute_in_current_hour = time_now - day*24*60 - hour*60

    # Determine how many minutes there is left in the current and next hour based on the time horizon
    minutes_current_hour = min(60-minute_in_current_hour , planning_horizon)
    minutes_next_hour = planning_horizon - minutes_current_hour
    
    # Calculate the net demand for the current and next hour
    net_demand_current = station.get_arrive_intensity(day, hour) - station.get_leave_intensity(day, hour)
    net_demand_next = station.get_arrive_intensity(day, hour+1) - station.get_leave_intensity(day, hour+1)
    
    # Calculate the average net demand for the planning horizon
    net_demand_planning_horizon = (minutes_current_hour*net_demand_current + 
                                   minutes_next_hour*net_demand_next) / (planning_horizon if planning_horizon != 0 else 1)
    
    return round(net_demand_planning_horizon)


def calculate_hourly_discharge_rate(simul, total_num_bikes_in_system, is_electric = True, is_station_based = False):
    """
    Returns average battery discharge over the whole system.
    TODO return 0 if there are no E-bikes

    Parameters:
    - simul = Simulator
    - total_num_bikes_in_system = Total number of bikes in the system
    - is_electric = boolean, only checking discharge for electric systems
    """

    if not is_electric:
        return 1 #TODO

    time_now = simul.time
    day = simul.day()
    hour = simul.hour()
    next_hour = (hour + 1) % 24
    next_day = day if next_hour > hour else day +1

    locations = simul.state.get_stations() if is_station_based else simul.state.get_areas()
    
    number_of_trips_current_hour = sum([loc.get_leave_intensity(day, hour) for loc in locations])
    number_of_trips_next_hour = sum([loc.get_leave_intensity(next_day, next_hour) for loc in locations])

    minutes_remaining = 60 - (time_now % 60)
    number_of_trips_next_60_min = (minutes_remaining * number_of_trips_current_hour + (60 - minutes_remaining) * number_of_trips_next_hour) / 60

    total_system_battery_discharge = number_of_trips_next_60_min * AVERAGE_LENGHT_OF_TRIP * BATTERY_CHANGE_PER_MINUTE

    return total_system_battery_discharge / total_num_bikes_in_system

def copy_arr_iter(arr):
    root = []
    stack = [(arr,root)]
    while stack:
        (o,d), *stack = stack
        assert isinstance(o, list)
        for i in o:
            if isinstance(i, list):
                p = (i, [])
                d.append(p[1])
                stack.append(p)
            else:
                d.append(i)
    return root


def generate_discounting_factors(nVisits, end_factor = 0.1):
    """
    Returns a list of nVisits scores from 1 to 0, of how much each visit is supposed to count on the total score.

    Parameters:
    - nVisits = Number of visits the vehicle makes
    - end_factor = The smallest weight for a visit to count
    """
    discounting_factors = []
    len = nVisits
    rate = (1/end_factor)**(1/len)-1
    for visit in range(0,len):
        discount_factor = 1/((1+rate)**visit)
        discounting_factors.append(discount_factor)
    return discounting_factors