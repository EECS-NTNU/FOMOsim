"""
This file contains a greedy policy
"""
import copy

from policies import Policy

from policies.gleditsch_hagen.utils import calculate_net_demand, calculate_time_to_violation
from policies.criticality_scores import calculate_criticality_normalized
from settings import *

import sim
# import abc

# import init_state
#import init_state.entur.methods
#import init_state.entur.scripts

class GreedyPolicy(Policy):
    def __init__(self,
                 crit_weights=[0.1,0.2,0.3,0.4], 
                 cutoff=0.3, 
                 service_hours=None,
                 swap_threshold = BATTERY_LIMIT_TO_SWAP,
                 ):   #[0,0,0,1] for deviation from target state
        
        self.swap_threshold = swap_threshold
        super().__init__()

        if service_hours is not None:
            self.set_time_of_service(service_hours[0],service_hours[1])
            
        #Two options for :
        # 1. Aim for target state THIS SHOULD BE THE STANDARD
        # 2. Maximum pick-up or delivery amount (somewhat of a special case of target state)
        
        #Then, next station can be based on
        # 1. Criticality Score
        # 2. Deviation from target state  (this is a special case of criticality score)
        
        #- WEIGHTS           
        self.crit_weights = crit_weights
        [self.omega1,self.omega2,self.omega3,self.omega4] = crit_weights # time_to_violation, net_demand, driving_time, deviation_target_state
                    
        self.cutoff = cutoff       # to decide when to go the pickup or delivery station next


    def get_best_action(self, state, vehicle):
        
        batteries_to_swap = []
        bikes_to_pickup = []
        bikes_to_deliver = []
            

        #########################################
        #               WHAT TO DO              #
        #########################################
        
        
        num_bikes_vehicle = len(vehicle.get_bike_inventory())
        
        number_of_bikes_to_pick_up = 0
        number_of_bikes_to_deliver = 0
        number_of_batteries_to_swap = 0

        if not vehicle.is_at_depot():
            target_state = round(vehicle.location.get_target_state(state.day(), state.hour()))
            num_bikes_station = len(vehicle.location.bikes)
            if num_bikes_station < target_state: #deliver bikes
                
                #deliver bikes, max to the target state
                number_of_bikes_to_deliver = min(num_bikes_vehicle,target_state-num_bikes_station)
                #bikes_to_deliver = [bike.id for bike in vehicle.get_bike_inventory()[:number_of_bikes_to_deliver]]
                bikes_to_deliver = [bike.bike_id for bike in vehicle.get_bike_inventory()[:number_of_bikes_to_deliver]]
                bikes_to_pickup = []
                number_of_bikes_to_pick_up = 0
                
                # Swap as many bikes (or batteries!) as possible as this station most likely needs it
                # TO DO: MAKE SURE THIS MAKES SENSE!!!!
                swappable_bikes = vehicle.location.get_swappable_bikes()
                number_of_batteries_to_swap = min(vehicle.battery_inventory, len(swappable_bikes))
                #batteries_to_swap = [bike.id for bike in swappable_bikes][:number_of_batteries_to_swap]
                batteries_to_swap = [bike.bike_id for bike in swappable_bikes][:number_of_batteries_to_swap]
                
            elif num_bikes_station > target_state: #pick-up bikes
            
                number_of_bikes_to_deliver = 0
                bikes_to_deliver = []
                remaining_vehicle_capacity = vehicle.bike_inventory_capacity - len(vehicle.bike_inventory)
                number_of_bikes_to_pick_up = min(num_bikes_station-target_state,remaining_vehicle_capacity)
                #print(vehicle.location.bikes.values())
                #bikes_to_pickup = [bike.id for bike in vehicle.location.bikes.values()][:number_of_bikes_to_pick_up]
                bikes_to_pickup = [bike.bike_id for bike in vehicle.location.bikes.values()][:number_of_bikes_to_pick_up]

                # UPDATE/ TODO Do not swap any bikes/batteries in a station with a lot of bikes
                batteries_to_swap = []
                number_of_batteries_to_swap = 0
            
            else: #num bikes is exactly at target state
                bikes_to_deliver = []
                bikes_to_pickup = []
                number_of_bikes_to_pick_up = 0
                number_of_batteries_to_swap = 0
                batteries_to_swap = []

        ##########################################
        #               WHERE TO GO              #
        ##########################################

        # If vehicles has under 10% battery inventory, go to depot.
        if ((vehicle.battery_inventory - number_of_batteries_to_swap < vehicle.battery_inventory_capacity * 0.1 )
            # - number_of_bikes_to_pick_up  #why should this impact the battery stock... ?? TODO check
            and not vehicle.is_at_depot() and (len(state.depots) > 0)):
            next_location_id = list(state.depots.values())[0].id  # TO DO: go to nearest depot, not just the first
        else:
            #FILTERING
            #tabu_list = [vehicle2.location.id for vehicle2 in state.vehicles] #do not go where other vehicles are (going)
            tabu_list = [state.vehicles[vehicle2].location.id for vehicle2 in state.vehicles] #do not go where other vehicles are (going)


            #potential_stations = [station for station in state.locations if station.id not in tabu_list]
            potential_stations = [station for station in state.locations if station not in tabu_list]

            #net_demands = {station.id:calculate_net_demand(station,state.time,state.day(),state.hour(),planning_horizon=60) 
            #                for station in potential_stations}
            # target_states = {station.id:station.get_target_state(state.day(), state.hour()) 
            #                     for station in potential_stations}
            # driving_times = {station.id:state.get_vehicle_travel_time(vehicle.location.id,station.id) 
            #                     for station in potential_stations}
            # times_to_violation = {station.id:calculate_time_to_violation(net_demands[station.id],station) 
            #                     for station in potential_stations}
            # deviations_from_target_state = {station.id:abs(target_states[station.id]-len(station.bikes))
            #                     for station in potential_stations}
            # potential_pickup_stations = [station for station in potential_stations if 
            #                                 station.number_of_bikes() + net_demands[station.id] > target_states[station.id]]
            # potential_delivery_stations = [station for station in potential_stations if 
            #                                 station.number_of_bikes() + net_demands[station.id] < target_states[station.id]]
            net_demands = {station:calculate_net_demand(state.locations[station],state.time,state.day(),state.hour(),planning_horizon=60) 
                            for station in potential_stations}
            target_states = {station:state.locations[station].get_target_state(state.day(), state.hour()) 
                            for station in potential_stations}
            driving_times = {station:state.get_vehicle_travel_time(vehicle.location.id,station) 
                                for station in potential_stations}
            times_to_violation = {station:calculate_time_to_violation(net_demands[station],state.locations[station]) 
                                for station in potential_stations}
            deviations_from_target_state = {station:abs(target_states[station]-len(state.locations[station].bikes))
                                for station in potential_stations}
            potential_pickup_stations = [station for station in potential_stations if 
                                            state.locations[station].number_of_bikes() + net_demands[station] > target_states[station]]
            potential_delivery_stations = [station for station in potential_stations if 
                                            state.locations[station].number_of_bikes() + net_demands[station] < target_states[station]]
            
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
            criticalities = {station:calculate_criticality_normalized(
                self.crit_weights,
                state,
                vehicle.location.id,
                station,
                net_demands[station],
                max([abs(x) for x in list(net_demands.values())]),
                max(list(driving_times.values())),
                max(list(times_to_violation.values())),
                max(list(deviations_from_target_state.values()))
                ) for station in potential_stations}                                                    
            
            #sort to get the most promising ones
            criticalities = dict(sorted(criticalities.items(), key=lambda item: item[1], reverse=True)) #descending order
            
            #pick the best
            if len(criticalities)==0:
                # commented out by Lasse since it clutters the simulation output
                # print('no stations with non-zero criticality, route to random station')
                # print('problem seems to be that target state is empty... ??')
                potential_stations2 = [station for station in state.locations if station.id not in tabu_list]
                next_location_id = state.rng.choice(potential_stations2).id
            else: 
                next_location_id = list(criticalities.keys())[0]
    
        return sim.Action(
            batteries_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_location_id,
        )
    
        
       
    
    
