"""
This file contains a greedy policy
"""
import copy

from policies import Policy
import sim
# import abc
import random


# import init_state
import init_state.entur.methods
import init_state.entur.scripts

class GreedyPolicyOld(Policy):
    def __init__(self):
        super().__init__()

        self.set_time_of_service()

    def get_best_action(self, simul, vehicle):
        
        
        bikes_to_swap = []
        bikes_to_pickup = []
        bikes_to_deliver = []
        
        #TIME OF SERVICE, TO DO: send to depot
        if  self.hour_from <= simul.hour() < self.hour_to: 
        
            vehicle_has_bike_inventory = len(vehicle.bike_inventory) > 0
            if vehicle.is_at_depot():
                bikes_to_deliver = []
                bikes_to_pickup = []
                number_of_bikes_to_pick_up = 0
                number_of_bikes_to_swap = 0
                bikes_to_swap = []
            else:
                # If vehicle has bike inventory, deliver all bikes and swap all swappable bikes
                if vehicle_has_bike_inventory:
                    # Deliver all bikes in bike inventory, and don't pick up any new bikes
                    bikes_to_deliver = [
                        bike.id for bike in vehicle.get_bike_inventory()
                    ]
                    bikes_to_pickup = []
                    number_of_bikes_to_pick_up = 0
                    # Swap as many bikes as possible as this station most likely needs it
                    swappable_bikes = vehicle.location.get_swappable_bikes()
                    number_of_bikes_to_swap = min(
                        vehicle.battery_inventory, len(swappable_bikes)
                    )
                    bikes_to_swap = [bike.id for bike in swappable_bikes][
                        :number_of_bikes_to_swap
                    ]
                else:
                    # Pick up as many bikes as possible, the min(bike capacity, deviation from ideal state)
                    number_of_bikes_to_pick_up = int(max(
                        min(
                            vehicle.bike_inventory_capacity
                            - len(vehicle.bike_inventory),
                            vehicle.battery_inventory,
                            len(vehicle.location.bikes)
                            - vehicle.location.get_target_state(simul.day(), simul.hour()),
                        ),
                        0,
                    ))
                    bikes_to_pickup = list(vehicle.location.bikes.keys())[:number_of_bikes_to_pick_up]
                    # Do not swap any bikes in a station with a lot of bikes
                    bikes_to_swap = []
                    number_of_bikes_to_swap = 0
                    # There are no bikes to deliver due to empty inventory
                    bikes_to_deliver = []
    
            def get_next_location_id(simul, is_finding_positive_deviation):
                tabu_list = [ vehicle.location.id for vehicle in simul.state.vehicles ]
    
                return sorted(
                    [
                        station
                        for station in simul.state.stations.values()
                        if station.id not in tabu_list
                    ],
                    key=lambda station: len(station.get_available_bikes())
                    - station.get_target_state(simul.day(), simul.hour()),
                    reverse=is_finding_positive_deviation,
                )[0].id
    
            # If vehicles has under 10% battery inventory, go to depot.
            if (
                vehicle.battery_inventory
                - number_of_bikes_to_swap
                - number_of_bikes_to_pick_up
                < vehicle.battery_inventory_capacity * 0.1
            ) and not vehicle.is_at_depot() and (len(simul.state.depots) > 0):
                next_location_id = list(simul.state.depots.values())[0].id
            else:
                """
                If vehicle has bike inventory upon arrival,
                go to new positive deviation station to pick up new bikes.
                If there are no bike inventory, go to station where you
                can drop off bikes picked up in this station, ergo negative deviation station.
                If, however, you are in the depot, you should do the opposite as the depot does not
                change the bike inventory.
                """
                visit_positive_deviation_station_next = (len(vehicle.bike_inventory) + len(bikes_to_pickup) - len(bikes_to_deliver)) <= 0
                next_location_id = get_next_location_id(simul, visit_positive_deviation_station_next)

        else:
            next_location_id = random.choice([station.id for station in simul.state.stations.values()])  #  

        return sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_location_id,
        )
