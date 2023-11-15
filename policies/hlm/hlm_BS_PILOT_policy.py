from policies import Policy
import sim
import Visit
import Plan

import numpy as np
import time

########################################## 
#         BS_PILOT - policy class        #
##########################################

class BS_PILOT(Policy): #Add default values from sepeate setting sheme
    def __init__(self, max_depth, number_of_successors, time_horizon, criticality_weights_sets, evaluation_weights, number_of_scenarios, discounting_factor):
        self.max_depth = max_depth
        self.number_of_successors = number_of_successors
        self.time_horizon = time_horizon
        self.criticality_weights_set = criticality_weights_sets
        self.evaluation_weights = evaluation_weights
        self.number_of_scenarios = number_of_scenarios
        self.discounting_factor = discounting_factor
        super().__init__()

    ###############################################
    #  Returns an action element that contains:   #
    #  * Number of batteries to swap              #
    #  * Number of e-scooters to pick up          #
    #  * Number of e-scooters to deliver          #
    #  * Next station to visit                    #
    #                                             #
    # This is how our code talks to the simulator #            
    ###############################################
    
    def get_best_action(self, simul, vehicle):
        start_logging_time = time.time() #Hva brukes dette til?
        next_station = None
        escooters_to_pickup = []
        escooters_to_deliver = []

        end_time = simul.time + self.time_horizon #Vet ikke hva denne brukes til heller.

        #########################################################################################
        #   Number of bikes to pick up / deliver is choosen greedy based on clusters in reach   #
        #   Which bike ID´s to pick up / deliver is choosen based on battery level              #   
        #   How many batteries to swap choosen based on battery inventory and status on station #           
        #########################################################################################

        escooters_to_pickup, escooters_to_deliver, batteries_to_swap = calculate_loading_quantities_and_swaps_greedy(vehicle, simul, vehicle.location)
        number_of_escooters_pickup = len(escooters_to_pickup)
        number_of_escooters_deliver = len(escooters_to_deliver)
        number_of_batteries_to_swap = len(batteries_to_swap)


        ####################################################################
        #  if eta?? == 0 keep greedy quantity values                       #
        #  Else find new quantities with future demand into consideration  #
        ####################################################################

        plan_dict = dict()
        for v in simul.state.vehicles:
            if v.eta == 0:
                plan_dict[v.id] = [Visit(v.location, number_of_escooters_pickup, number_of_escooters_deliver, number_of_batteries_to_swap, simul.time, v)]
            else:
                number_of_escooters_pickup, number_of_escooters_deliver, number_of_batteries_to_swap = self.calculate_loading_quantities_and_swaps_pilot(v, simul, v.location, v.eta)
                plan_dict[v.id] = [Visit(v.location, int(number_of_escooters_pickup), int(number_of_escooters_deliver), int(number_of_batteries_to_swap), v.eta, v)]
        
        tabu_list = [v.location.id for v in simul.state.vehicles]
        plan = Plan(plan_dict, tabu_list)

    

    def calculate_loading_quantities_and_swaps_pilot(self, vehicle, simul, station, current_time):
        escooters_in_veichle = vehicle.get_bike_inventory()

        number_of_escooters_pickup = 0
        number_of_escooters_deliver = 0
        number_of_escooters_swap = 0

        














#############################################################################################
#   Number of bikes to pick up / deliver is choosen greedy based on clusters in reach       #
#   Which bike ID´s to pick up / deliver is choosen based on battery level                  #
#   How many and which escooters to swap battery on based on inventory and station status   #                                      
#   Applied functionality such that scooters with battery level < thershold does not count  #             
#############################################################################################

def calculate_loading_quantities_and_swaps_greedy(vehicle, simul, station):
    num_escooters_vehicle = len(vehicle.get_bike_inventory())

    target_state = round(station.get_target_state(simul.day(), simul.hour())) #Denne må vi finne ut hvordan lages
    num_escooters_station = station.number_of_bikes()
    num_escooters_accounted_for_battery_swaps = num_escooters_accounted_for_battery_swaps(station, num_escooters_station, vehicle)

    ################################################################
    #  Adjusting numbers based on status at neighboring stations   #
    #  Changed from capacity to diviation from ideal state         #
    ################################################################

    starved_neighbors = 0
    overflowing_neighbors = 0

    for neighbor in station.neighboring_stations:
        num_escooters_neighbor = neighbor.number_of_bikes() - len(neighbor.get_swappable_bikes(20)) #Bytt ut med et dokument tilsvarende settings
        neighbor_target_state = round(neighbor.get_target_state(simul.day(), simul.hour()))
        if num_escooters_neighbor < 0.1 * neighbor_target_state:
            starved_neighbors += 1
        elif num_escooters_neighbor > 0.9 * neighbor_target_state:
            overflowing_neighbors += 1

    #######################################################################
    #  Find out whether the station is a pick-up or a deliver station     #
    #  And based on that the quantities for pickup, deliveries and swaps  #
    #######################################################################

    if num_escooters_accounted_for_battery_swaps < target_state: #Ta hensyn til nabocluster her?
        number_of_escooters_to_deliver = min(num_escooters_vehicle, target_state - num_escooters_accounted_for_battery_swaps + 2*starved_neighbors) # discuss 2*starved_neighbors part, ta hensyn til postensielle utladede scootere i bilen
        escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = id_escooters_accounted_for_battery_swaps(station, vehicle, number_of_escooters_to_deliver, "deliver")
        escooters_to_pickup_accounted_for_battery_swaps = []
    
    elif num_escooters_accounted_for_battery_swaps > target_state:
        remaining_cap_vehicle = vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory())
        number_of_escooters_to_pickup = min(remaining_cap_vehicle, num_escooters_accounted_for_battery_swaps - target_state + 2*overflowing_neighbors) #discuss logic behind this
        escooters_to_deliver_accounted_for_battery_swaps=[]
        escooters_to_pickup_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = id_escooters_accounted_for_battery_swaps(station, vehicle, number_of_escooters_to_pickup, "pickup")

    else:
        escooters_to_pickup_accounted_for_battery_swaps = []
        escooters_to_deliver_accounted_for_battery_swaps = []

        escooters_in_station_low_battery = station.get_swappable_bikes(20)
        num_escooters_to_swap = min(len(escooters_in_station_low_battery),vehicle.battery_inventory)
        escooters_to_swap_accounted_for_battery_swap = escooters_in_station_low_battery[:num_escooters_to_swap]


    
    return escooters_to_pickup_accounted_for_battery_swaps, escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap




#######################################################################################################################
# Simple calculation function to take low battery level into consideration when choosing number to deliver or pickup  #
#######################################################################################################################

def num_escooters_accounted_for_battery_swaps(station, num_escooters_station, vehicle): 
    return num_escooters_station - len(station.get_swappable_bikes(20)) + min(station.get_swappable_bikes(20),vehicle.battery_inventory)


############################################################################################
# If deliver, swap as many low battries in station as possible then deliver rest of bikes  #
# Swaps from low battry until threshold, default 20                                        #
# Delivers from high battery until threshold                                               #  
#                                                                                          #
# If pickup, pickup and swap as many escooters with low battry as possible                 #
# Then pickup rest from high battry end or swap remaining swaps                            #
############################################################################################

def id_escooters_accounted_for_battery_swaps(station, vehicle, number_of_escooters, type):
    escooters_in_station = sorted(station.bikes.values(), key=lambda bike: bike.battery, reverse=False)
    escooters_in_vehicle =  sorted(vehicle.get_bike_inventory(), key=lambda bike: bike.battery, reverse=False) 

    if type == "deliver":
        number_of_escooters_to_swap = min(len(station.get_swappable_bikes(20)),vehicle.battery_inventory)
        number_of_escooters_to_deliver = number_of_escooters

        escooters_to_swap = [escooter.id for escooter in escooters_in_station][:number_of_escooters_to_swap]
        escooters_to_deliver = [escooter.id for escooter in escooters_in_vehicle][-number_of_escooters_to_deliver:]

        return escooters_to_deliver, escooters_to_swap
    
    elif type == "pickup": 
        number_of_escooters_to_swap_and_pickup = min(len(station.get_swappable_bikes(70)),vehicle.battery_inventory,number_of_escooters) # decide threshold for swap when pickup
        number_of_escooters_to_only_pickup = number_of_escooters - number_of_escooters_to_swap_and_pickup
        number_of_escooters_to_only_swap = max(0,len(station.get_swappable_bikes(20)) - number_of_escooters_to_swap_and_pickup)

        escooters_to_swap = []
        escooters_to_pickup = [escooter.id for escooter in escooters_in_station][:number_of_escooters_to_swap_and_pickup]
        if number_of_escooters_to_only_pickup > 0:
            escooters_to_pickup += escooters_in_station[-number_of_escooters_to_only_pickup:]
        
        elif number_of_escooters_to_only_swap > 0:
            escooters_to_swap += escooters_in_station[number_of_escooters_to_swap_and_pickup:number_of_escooters_to_swap_and_pickup+number_of_escooters_to_only_swap]

        return escooters_to_pickup, escooters_to_swap
    
    return [],[]











    






       
    
    




