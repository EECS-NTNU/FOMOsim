import sim
from sim import Event
from settings import *
import numpy as np
import random

class EScooterDeparture(Event):
    """
    Event fired when a customer requests a trip from a given departure area. Creates a Lost Trip or Bike Arrival
    event based on the availability of the area
    """

    def __init__(self, departure_time, departure_area_id):
        super().__init__(departure_time)
        self.departure_area_id = departure_area_id

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        super().perform(world)

        # get departure area
        departure_area = world.state.get_location_by_id(self.departure_area_id)

        # get all available bike in the area
        available_escooters = departure_area.get_available_bikes()

        # if there are no more available bikes -> make a LostTrip event for that departure time
        if len(available_escooters) > 0:
            escooter = available_escooters.pop(0)

            if FULL_TRIP:
                if world.state.rng.random() < RANDOM_DESTINATION_PROB:
                    # Exclude the current area from the random selection
                    other_areas = [area for area in world.state.get_areas() if area.location_id != self.departure_area_id]
                    arrival_area = world.state.rng.choice(other_areas)
                else:
                    # get an arrival area from the leave prob distribution
                    p=departure_area.get_move_probabilities(world.state, world.day(), world.hour())
                    sum_p = sum(p.values())
                    p_normalized = []
                    for i in p.keys():
                        if sum_p > 0:
                            p_normalized.append(p[i] * (1.0/sum_p))
                        else:
                            p_normalized.append(1/len(p))
                    arrival_area = world.state.rng.choice(world.state.get_areas(), p = p_normalized)

                # calculate arrival time
                travel_time = world.state.get_travel_time(
                    departure_area.location_id,
                    arrival_area.location_id,
                )

                # create an arrival event for the departed bike
                world.add_event(
                    sim.EScooterArrival(
                        self.time,
                        travel_time,
                        escooter,
                        arrival_area.location_id,
                        departure_area.location_id,
                    )
                )
                
            # remove bike from the departure area
            departure_area.remove_bike(escooter)

            world.state.set_bike_in_use(escooter)

            world.metrics.add_aggregate_metric(world, "escooter departure", 1)
            world.metrics.add_aggregate_metric(world, "events", 2)

        else:
            if FULL_TRIP:

                # Find closest area within allowed radius of Hexagons to search for available bikes
                closest_neighbour_with_bikes = world.state.get_closest_available_area(departure_area)

                if closest_neighbour_with_bikes:
                    # Find distance to closest neighbor and 
                    distance = departure_area.distance_to(closest_neighbour_with_bikes.get_lat(), closest_neighbour_with_bikes.get_lon())
                    
                    # Get probability distribution for finding arrival area
                    p = departure_area.get_move_probabilities(world.state, world.day(), world.hour())
                    sum_p = sum(p.values())
                    p_normalized = []
                    for i in p.keys():
                        if sum_p > 0:
                            p_normalized.append(p[i] * (1.0/sum_p)) # TODO, not sure if this is needed
                        else:
                            p_normalized.append(1/len(p))
                else:
                    distance = float('inf')

                # Use acceptance/rejection function to decide if roaming is accepted
                if self.acceptance_rejection(distance, world):
                    available_escooters = closest_neighbour_with_bikes.get_available_bikes()
                    escooter=available_escooters.pop(0)
                    
                    if world.state.rng.random() < RANDOM_DESTINATION_PROB:
                        # Exclude the current area from the random selection
                        other_areas = [area for area in world.state.get_areas() if area.location_id != self.departure_area_id]
                        arrival_area = world.state.rng.choice(other_areas)
                    else:
                        arrival_area = world.state.rng.choice(world.state.get_areas(), p = p_normalized)

                    # calculate arrival time 
                    #total travel time, roaming for bike from departure area to neighbour + cycling to arrival area
                    travel_time = world.state.get_travel_time(
                        closest_neighbour_with_bikes.location_id,
                        arrival_area.location_id,) + world.state.get_travel_time(departure_area.location_id,
                        closest_neighbour_with_bikes.location_id)*(BIKE_SPEED/WALKING_SPEED)

                    # remove bike from the new departure area
                    closest_neighbour_with_bikes.remove_bike(escooter)

                    world.state.set_bike_in_use(escooter)

                    # create an arrival event for the roaming user from the new departure area
                    world.add_event(
                        sim.EScooterArrival(
                            self.time,
                            travel_time,
                            escooter,
                            arrival_area.location_id,
                            closest_neighbour_with_bikes.location_id,
                        )
                    )

                    world.metrics.add_aggregate_metric(world, "escooter departure", 1)
                    world.metrics.add_aggregate_metric(world, "events", 2)

                    world.metrics.add_aggregate_metric(world, "roaming for escooters", 1)
                    world.metrics.add_aggregate_metric(world, "roaming distance for escooters", distance)

                else:
                    if departure_area.number_of_bikes() <= 0:
                        world.metrics.add_aggregate_metric(world, "escooter starvations", 1)
                    else:
                        world.metrics.add_aggregate_metric(world, "battery starvations", 1)

                    world.metrics.add_aggregate_metric(world, "events", 1)
                    world.metrics.add_aggregate_metric(world, "starvations", 1)
                    world.metrics.add_aggregate_metric(world, "failed events", 1)

        world.metrics.add_aggregate_metric(world, "trips", 1)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, departing from area {self.departure_area_id}>"
    
    def acceptance_rejection(self,distance, world):
        prob_acceptance = -1.6548*distance**2-0.7036*distance+1.0133
        random_roaming_limit = world.state.rng.uniform(0,1)
        if random_roaming_limit <= prob_acceptance:
            return True
        else:
            return False 