# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:34:40 2022

@author: steffejb
"""


def calculate_time_to_violation(net_demand,station):  #MOVE SOMEWHERE ELSE, seems to work when tested
    
    time_to_violation = 0
    if net_demand > 0:
        time_to_violation = (station.capacity - station.number_of_bikes() ) / net_demand
    elif net_demand < 0:
        time_to_violation = - station.number_of_bikes() / net_demand
    elif net_demand == 0:
        time_to_violation = 60*5  #No demand, then no violation in the next couple of hours (5) HARDCODING
    return time_to_violation


def calculate_net_demand(station,time_now, day,hour, planning_horizon):  #MOVE SOMEWHERE ELSE, seems to work when tested
    if planning_horizon > 60:
        print('not yet supported')
    
    minute_in_current_hour = time_now-day*24*60-hour*60 
    
    minutes_current_hour = min(60-minute_in_current_hour,planning_horizon)
    minutes_next_hour = planning_horizon - minutes_current_hour
    
    #NET DEMAND FOR LOCKS
    net_demand_current = station.get_arrive_intensity(day,hour) - station.get_leave_intensity(day,hour)
    net_demand_next = station.get_arrive_intensity(day,hour+1) - station.get_leave_intensity(day,hour+1)
    
    net_demand_planning_horizon = (minutes_current_hour*net_demand_current + 
                                   minutes_next_hour*net_demand_next)/planning_horizon
    
    return net_demand_planning_horizon


def extract_N_best_elements(a_list,N):
    best_elements = []
    iterate_over = min(N,len(a_list))
    for i in range(iterate_over):
        max_value = max(a_list)
        max_index = a_list.index(max_value)
        best_elements.append(max_index)
        a_list[max_index] = -1000000
    return best_elements
        
        