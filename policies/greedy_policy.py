"""
This file contains a greedy policy
"""
import copy

from policies import Policy

from policies.gleditsch_hagen.utils import calculate_net_demand, calculate_time_to_violation

import sim
# import abc
import random

# import init_state
import init_state.entur.methods
import init_state.entur.scripts

class GreedyPolicy(Policy):
    def __init__(self):
        super().__init__()

        #Two options for :
        # 1. Aim for target state THIS SHOULD BE THE STANDARD
        # 2. Maximum pick-up or delivery amount (somewhat of a special case of target state)
        
        #Then, next station can be based on
        # 1. Criticality Score
        # 2. Deviation from target state  (this is a special case of criticality score)

        #DEFINE SOME MORE PROPERTIES HERE!
        
        #- WEIGHTS
        
        
        #- Choice of criticality measure
        self.criticality_measure = 'weighted_average'  # OR 'weighted_average'
        if self.criticality_measure == 'deviation_from_target_state':
            self.omega1 = 0     # time_to_violation
            self.omega2 = 0     # net_demand
            self.omega3 = 0     # driving_time
            self.omega4 = 1     # deviation_from_target_state   (this was deviation when not visited)
        elif self.criticality_measure == 'weighted_average':
            self.omega1 = 0.1   # time_to_violation
            self.omega2 = 0.5   # net_demand
            self.omega3 = 0.1   # driving_time
            self.omega4 = 0.3   # deviation_from_target_state
 
        self.cutoff = 0.3       # to decide when to go the pickup or delivery station next


    def get_best_action(self, simul, vehicle):
        

        #########################################
        #               WHAT TO DO              #
        #########################################
        
        
        num_bikes_vehicle = len(vehicle.get_bike_inventory())

        bikes_to_deliver = []
        bikes_to_pickup = []
        number_of_bikes_to_pick_up = 0
        number_of_bikes_to_deliver = 0
        number_of_bikes_to_swap = 0
        bikes_to_swap = []
        if not vehicle.is_at_depot():
            target_state = round(vehicle.location.get_target_state(simul.day(), simul.hour()))
            num_bikes_station = len(vehicle.location.bikes)
            if num_bikes_station < target_state: #deliver bikes
                
                #deliver bikes, max to the target state
                number_of_bikes_to_deliver = min(num_bikes_vehicle,target_state-num_bikes_station)
                bikes_to_deliver = [bike.id for bike in vehicle.get_bike_inventory()[:number_of_bikes_to_deliver]]
                bikes_to_pickup = []
                number_of_bikes_to_pick_up = 0
                
                # Swap as many bikes (or batteries!) as possible as this station most likely needs it
                # TO DO: MAKE SURE THIS MAKES SENSE!!!!
                swappable_bikes = vehicle.location.get_swappable_bikes()
                number_of_bikes_to_swap = min(vehicle.battery_inventory, len(swappable_bikes))
                bikes_to_swap = [bike.id for bike in swappable_bikes][:number_of_bikes_to_swap]
                
            elif num_bikes_station > target_state: #pick-up bikes
            
                number_of_bikes_to_deliver = 0
                bikes_to_deliver = []
                remaining_vehicle_capacity = vehicle.bike_inventory_capacity - len(vehicle.bike_inventory)
                number_of_bikes_to_pick_up = min(num_bikes_station-target_state,remaining_vehicle_capacity)
                bikes_to_pickup = [bike.id for bike in vehicle.location.bikes.values()][:number_of_bikes_to_pick_up]
            
                # UPDATE/ TODO Do not swap any bikes/batteries in a station with a lot of bikes
                bikes_to_swap = []
                number_of_bikes_to_swap = 0
            
            else: #num bikes is exactly at target state
                bikes_to_deliver = []
                bikes_to_pickup = []
                number_of_bikes_to_pick_up = 0
                number_of_bikes_to_swap = 0
                bikes_to_swap = []

        ##########################################
        #               WHERE TO GO              #
        ##########################################

        # If vehicles has under 10% battery inventory, go to depot.
        if ((vehicle.battery_inventory - number_of_bikes_to_swap < vehicle.battery_inventory_capacity * 0.1 )
            # - number_of_bikes_to_pick_up  #why should this impact the battery stock... ?? TODO check
         and not vehicle.is_at_depot() and (len(simul.state.depots) > 0)):
            next_location_id = list(simul.state.depots.values())[0].id  # TO DO: go to nearest depot, not just the first
        else:
            tabu_list = [vehicle.location.id for vehicle in simul.state.vehicles]
            potential_stations = [station for station in simul.state.stations.values() if station.id not in tabu_list]
            net_demands = {station.id:calculate_net_demand(station,simul.time,simul.day(),simul.hour(),planning_horizon=60) 
                           for station in potential_stations}
            target_states = {station.id:station.get_target_state(simul.day(), simul.hour()) 
                             for station in potential_stations}
            
            potential_pickup_stations = [station for station in potential_stations if 
                                         station.number_of_bikes() + net_demands[station.id] > target_states[station.id]]
            potential_delivery_stations = [station for station in potential_stations if 
                                         station.number_of_bikes() + net_demands[station.id] < target_states[station.id]]
            
            bikes_at_vehicle_after_rebalancing = num_bikes_vehicle + number_of_bikes_to_pick_up - number_of_bikes_to_deliver
            cutoff = self.cutoff
            if cutoff*vehicle.bike_inventory_capacity <= bikes_at_vehicle_after_rebalancing  <= (1-cutoff)*vehicle.bike_inventory_capacity:
                potential_stations = potential_pickup_stations + potential_delivery_stations
            else:
                # During some hours (approx. 23:00-06:00), there is VERY little demand for bikes (system closed???)
                # , while there still is demand for locks (it shoud always be possible to return a bike). 
                # But during these hours, usually the net demand is zero.
                #THEREFORE, during the night, we allow all stations
                
                #In general, we need a better definition of what defines as a pickup or delivery station
                if bikes_at_vehicle_after_rebalancing <= cutoff*vehicle.bike_inventory_capacity:  #few bikes, so want to pickup
                    potential_stations = potential_pickup_stations
                elif bikes_at_vehicle_after_rebalancing >= (1-cutoff)*vehicle.bike_inventory_capacity: #want do deliver
                    potential_stations = potential_delivery_stations

            #calculate criticalities for potential stations
            criticalities = {station.id:self.calculate_criticality(simul,vehicle.location.id,station.id,net_demands[station.id],planning_horizon=60) 
                             for station in potential_stations}                                                    
            #sort to get the most promising ones
            criticalities = dict(sorted(criticalities.items(), key=lambda item: item[1], reverse=True)) #descending order
            
            #pick the best
            if len(criticalities)==0:
                print('no stations with non-zero criticality, route to random station')
                potential_stations2 = [station for station in simul.state.stations.values() if station.id not in tabu_list]
                next_location_id = random.choice(potential_stations2).id
            else: 
                next_location_id = list(criticalities.keys())[0]

        return sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_location_id,
        )
    
    
   
    def calculate_criticality(self,simul,start_station_id, potential_station_id,net_demand_ps,planning_horizon=60):  
        
        potential_station = simul.state.stations[potential_station_id]  
        
        driving_time = simul.state.get_vehicle_travel_time(start_station_id,potential_station_id)
        #net demand for LOCKS (so positive values is parking of bikes)
        net_demand = net_demand_ps
        #ALTERNATIVELY: use station.get_arrive_intensity(day,hour) and station.get_leave_intensity(day,hour)
        time_to_violation = calculate_time_to_violation(net_demand,potential_station)
        target_state = potential_station.get_target_state(simul.day(), simul.hour())
        deviation_from_target_state = abs(target_state-len(potential_station.bikes))
        
        return round((-self.omega1*time_to_violation + 
         self.omega2*net_demand -
         self.omega3*driving_time + 
        #self.omega4*deviation_not_visited     #THIS WAS DONE IN GleditschHagen
        self.omega4*deviation_from_target_state
        ),3)
    
    
    




#OLD CODE



# def get_next_location_id(simul, is_finding_positive_deviation):
#     tabu_list = [vehicle.location.id for vehicle in simul.state.vehicles ]
#     potential_stations = [station for station in simul.state.stations.values()
#         if station.id not in tabu_list]
#     return sorted(potential_stations, key=lambda station: len(station.get_available_bikes())
#         - station.get_target_state(simul.day(), simul.hour()),
#         reverse=is_finding_positive_deviation,
#     )[0].id




# """
# If vehicle has bike inventory upon arrival,
# go to new positive deviation station to pick up new bikes.
# If there are no bike inventory, go to station where you
# can drop off bikes picked up in this station, ergo negative deviation station.
# If, however, you are in the depot, you should do the opposite as the depot does not
# change the bike inventory.
# """
# visit_positive_deviation_station_next = (len(vehicle.bike_inventory) + len(bikes_to_pickup) - len(bikes_to_deliver)) <= 0
# next_location_id = get_next_location_id(simul, visit_positive_deviation_station_next)




# # If vehicle has bike inventory, deliver all bikes and swap all swappable bikes
# if vehicle_has_bike_inventory: #And station has available capacity
#     # Deliver all bikes in bike inventory, and don't pick up any new bikes
#     num_bikes_to_deliver = min(vehicle.get_bike_inventory(),)
#     bikes_to_deliver = [  #max number of bikes that can be delivered
#         bike.id for bike in vehicle.get_bike_inventory()
#     ]
#     bikes_to_pickup = []
#     number_of_bikes_to_pick_up = 0
    
#     # Swap as many bikes as possible as this station most likely needs it
#     swappable_bikes = vehicle.location.get_swappable_bikes()
#     number_of_bikes_to_swap = min(
#         vehicle.battery_inventory, len(swappable_bikes)
#     )
#     bikes_to_swap = [bike.id for bike in swappable_bikes][
#         :number_of_bikes_to_swap
#     ]
# else:
#     # Pick up as many bikes as possible, the min(bike capacity, deviation from ideal state)
#     number_of_bikes_to_pick_up = int(max(
#         min(
#             vehicle.bike_inventory_capacity
#             - len(vehicle.bike_inventory),
#             vehicle.battery_inventory,#why is this here???
#             len(vehicle.location.bikes)
#             - vehicle.location.get_target_state(simul.day(), simul.hour()),
#         ),
#         0,
#     ))
#     bikes_to_pickup = list(vehicle.location.bikes.keys())[:number_of_bikes_to_pick_up]
#     # Do not swap any bikes in a station with a lot of bikes
#     bikes_to_swap = []
#     number_of_bikes_to_swap = 0
#     # There are no bikes to deliver due to empty inventory
#     bikes_to_deliver = []