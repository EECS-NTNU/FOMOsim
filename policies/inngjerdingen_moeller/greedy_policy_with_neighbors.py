"""
This file contains a greedy policy
""" 
import numpy as np 
from policies import Policy
from policies.gleditsch_hagen.utils import calculate_net_demand
from policies.inngjerdingen_moeller.criticality_score_neighbor import calculate_criticality
import sim


class GreedyPolicyNeighborhoodInteraction(Policy):
    def __init__(self,crit_weights=[0.3,0.15,0.25,0.2,0.1], cutoff_vehicle=0.3, cutoff_station=0.15, service_hours=None):
        super().__init__()
        
        if service_hours is not None:
            self.set_time_of_service(service_hours[0],service_hours[1])
            
        #- WEIGHTS           
        self.crit_weights = crit_weights
    
        self.cutoff_vehicle = cutoff_vehicle       # to decide when to go the pickup or delivery station next
        self.cutoff_station = cutoff_station

    def get_best_action(self, simul, vehicle):
        
        batteries_to_swap = []
        bikes_to_pickup = []
        bikes_to_deliver = [] 
        

        #########################################
        #               WHAT TO DO              #
        #########################################
        
        num_bikes_vehicle = len(vehicle.get_bike_inventory())
        bikes_to_pickup, bikes_to_deliver = calculate_loading_quantities_greedy(vehicle, simul, vehicle.location)
        number_of_bikes_to_pick_up = len(bikes_to_pickup)
        number_of_bikes_to_deliver = len(bikes_to_deliver)
        
        bikes_at_vehicle_after_rebalancing = num_bikes_vehicle + number_of_bikes_to_pick_up - number_of_bikes_to_deliver
        ##########################################
        #               WHERE TO GO              #
        ##########################################
        tabu_list = [vehicle2.location.id for vehicle2 in simul.state.vehicles] #do not go where other vehicles are (going)
        potential_stations = find_potential_stations(simul, self.cutoff_vehicle, self.cutoff_station, vehicle, bikes_at_vehicle_after_rebalancing, tabu_list)

        #calculate criticalities for potential stations, sorted by criticality
        criticalities = calculate_criticality(self.crit_weights, simul, potential_stations, vehicle.location) # dict {station: criticality} sorted by crit.score                           
        
        #pick the best
        if len(criticalities)==0:
            potential_stations2 = [station for station in simul.state.locations if station.id not in tabu_list]
            
            rng_greedy = np.random.default_rng(None)
            next_location_id = rng_greedy.choice(potential_stations2).id
        else: 
            next_location_id = list(criticalities.keys())[0].id
    

        return sim.Action(
            batteries_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_location_id,
        )
    

def calculate_loading_quantities_greedy(vehicle, simul, station):
    num_bikes_vehicle = len(vehicle.get_bike_inventory())
    
    number_of_bikes_to_pick_up = 0
    number_of_bikes_to_deliver = 0

    target_state = round(station.get_target_state(simul.day(), simul.hour()))
    num_bikes_station = station.number_of_bikes()

    #can calculate a "neighborhood-demand" (negative or positive) and add this to the target state to compensate for neighborhood interactions 

    starved_neighbors = 0
    for neighbor in station.neighboring_stations:
        num_bikes_neighbor = neighbor.number_of_bikes()
        if num_bikes_neighbor < 0.1*neighbor.get_target_state():
            starved_neighbors += 1

    if num_bikes_station < target_state: #deliver bikes
        #deliver bikes, max to the target state
        number_of_bikes_to_deliver = min(num_bikes_vehicle, target_state - num_bikes_station + 2*starved_neighbors)
        bikes_to_deliver = [bike.id for bike in vehicle.get_bike_inventory()[:number_of_bikes_to_deliver]]
        bikes_to_pickup = []
        
    elif num_bikes_station > target_state: #pick-up bikes
        bikes_to_deliver = []
        remaining_vehicle_capacity = vehicle.bike_inventory_capacity - len(vehicle.bike_inventory)
        number_of_bikes_to_pick_up = min(num_bikes_station - target_state, remaining_vehicle_capacity)
        bikes_to_pickup = [bike.id for bike in station.bikes.values()][:number_of_bikes_to_pick_up]
    
    else: #num bikes is exactly at target state
        bikes_to_deliver = []
        bikes_to_pickup = []
    return bikes_to_pickup, bikes_to_deliver


def find_potential_stations(simul, cutoff_vehicle, cutoff_station, vehicle, bikes_at_vehicle, tabu_list):
    potential_stations = [station for station in simul.state.locations if station.id not in tabu_list]
    
    net_demands = {station.id:calculate_net_demand(station,simul.time,simul.day(),simul.hour(),planning_horizon=60) 
                    for station in potential_stations}
    target_states = {station.id:station.get_target_state(simul.day(), simul.hour()) 
                        for station in potential_stations}
    
    potential_pickup_stations = [station for station in potential_stations if 
                                    station.number_of_bikes() + net_demands[station.id] > (1+cutoff_station)*target_states[station.id]]
    potential_delivery_stations = [station for station in potential_stations if 
                                    station.number_of_bikes() + net_demands[station.id] < (1-cutoff_station)*target_states[station.id]]
    
    if cutoff_vehicle*vehicle.bike_inventory_capacity <= bikes_at_vehicle  <= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
        potential_stations = potential_pickup_stations + potential_delivery_stations
    else:
        if bikes_at_vehicle <= cutoff_vehicle*vehicle.bike_inventory_capacity:  #few bikes, so want to pickup
            potential_stations = potential_pickup_stations
        elif bikes_at_vehicle >= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity: #want do deliver
            potential_stations = potential_delivery_stations
    return potential_stations