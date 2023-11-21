from Variables import AVARAGE_LENGHT_OF_TRIP
from settings import BATTERY_CHANGE_PER_MINUTE

#####################################################################################################################################
# Gleditsch & Hagen fuction for calculating net demand - this is what inngerdingen og møller uses                                   #
# We need might need to consider batterylevel in get arrive_intensity                                                               #
# Planning horizon is 60 minutes - that means that this function does not support planning further into the future than 60 minutes  #
#####################################################################################################################################

def calculate_net_demand(station, time_now, day, hour, planning_horizon): 
    if planning_horizon > 60:
        print('not yet supported') #Is this a problem for us - betyr at man ikke kan planlegge lengre enn 60 minuttter frem i tid 
    
    minute_in_current_hour = time_now-day*24*60-hour*60 
    
    minutes_current_hour = min(60-minute_in_current_hour,planning_horizon)
    minutes_next_hour = planning_horizon - minutes_current_hour
    
    #NET DEMAND(I think we can use this as it is)
    net_demand_current = station.get_arrive_intensity(day,hour) - station.get_leave_intensity(day,hour)
    net_demand_next = station.get_arrive_intensity(day,hour+1) - station.get_leave_intensity(day,hour+1)
    
    net_demand_planning_horizon = (minutes_current_hour*net_demand_current + 
                                   minutes_next_hour*net_demand_next)/planning_horizon
    
    return 2*net_demand_planning_horizon #Returns demand pr hour *2??



#################################################################
# NEW FUNCTIONALITY                                             #
# Avarage battery discharge rate per hour over the whole system #
#################################################################

def calculate_hourly_discharge_rate(simul):
    total_num_bikes_in_system = sum(station.number_of_bikes() for station in simul.state.stations) + len(simul.state.bikes_in_use()) #flytt hvis lang kjøretid

    time_now = simul.time
    day = simul.day()
    hour = simul.hour()

    number_of_trips_current_hour = sum(station.get_arrive_intensity(day,hour) for station in simul.state.stations)
    number_of_trips_next_hour = sum(station.get_arrive_intensity(day, hour+1) for station in simul.state.stations)

    number_of_trips_next_60_min = min(60-(time_now-day*24*60-hour*60),60)*number_of_trips_current_hour/60 + (60 - (time_now-day*24*60-hour*60))*number_of_trips_next_hour/60

    return ((number_of_trips_next_60_min * AVARAGE_LENGHT_OF_TRIP) * BATTERY_CHANGE_PER_MINUTE) / total_num_bikes_in_system


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
                d.append(p)
    return root


def generate_discounting_factors(nVisits, end_factor = 0.1):
    discounting_factors = []
    len = nVisits
    rate = (1/end_factor)**(1/len)-1
    for visit in range(0,len):
        discount_factor = 1/((1+rate)**visit)
        discounting_factors.append(discount_factor)
    return discounting_factors




