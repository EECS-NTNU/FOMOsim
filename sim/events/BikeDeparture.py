import sim
from sim import Event
from settings import *
import numpy as np
import random

class BikeDeparture(Event):
    """
    Event fired when a customer requests a trip from a given departure station. Creates a Lost Trip or Bike Arrival
    event based on the availability of the station
    """

    def __init__(self, departure_time: int, departure_station_id: int):
        super().__init__(departure_time)
        self.departure_station_id = departure_station_id

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        super().perform(world)

        # get departure station
        departure_station = world.state.get_location_by_id(self.departure_station_id)

        # get all available bike in the station
        available_bikes = departure_station.get_available_bikes()

        # if there are no more available bikes -> make a LostTrip event for that departure time
        if len(available_bikes) > 0:
            bike = available_bikes.pop(0)

            if FULL_TRIP:
                # get an arrival station from the leave prob distribution

                p=departure_station.get_move_probabilities(world.state, world.day(), world.hour())
                sum = 0.0
                for i in range(len(p)):
                    sum += p[i]
                p_normalized = []
                for i in range(len(p)):
                    if sum > 0:
                        p_normalized.append(p[i] * (1.0/sum)) # TODO, not sure if this is needed
                    else:
                        p_normalized.append(1/len(p))
                arrival_station = world.state.rng.choice(world.state.locations, p = p_normalized)

                travel_time = world.state.get_travel_time(
                    departure_station.id,
                    arrival_station.id,
                )

                # calculate arrival time

                # create an arrival event for the departed bike
                world.add_event(
                    sim.BikeArrival(
                        self.time,
                        travel_time,
                        bike,
                        arrival_station.id,
                        departure_station.id,
                    )
                )

            # remove bike from the departure station
            departure_station.remove_bike(bike)

            world.state.bike_in_use(bike)

        else:
            if FULL_TRIP:
                closest_neighbour_with_bikes = world.state.get_neighbours(departure_station,1,not_empty=True)[0]
                distance = departure_station.distance_to(closest_neighbour_with_bikes.get_lat(), closest_neighbour_with_bikes.get_lon())
                p=departure_station.get_move_probabilities(world.state, world.day(), world.hour())
                sum = 0.0
                for i in range(len(p)):
                    sum += p[i]
                p_normalized = []
                for i in range(len(p)):
                    if sum > 0:
                        p_normalized.append(p[i] * (1.0/sum)) # TODO, not sure if this is needed
                    else:
                        p_normalized.append(1/len(p))
                if self.acceptance_rejection(distance):
                    available_bikes = closest_neighbour_with_bikes.get_available_bikes()
                    bike=available_bikes.pop(0)
                    
                    arrival_station = world.state.rng.choice(world.state.locations, p = p_normalized)

                    travel_time = world.state.get_travel_time(
                        closest_neighbour_with_bikes.id,
                        arrival_station.id,) + world.state.get_travel_time(departure_station.id,
                        closest_neighbour_with_bikes.id)*(BIKE_SPEED/WALKING_SPEED) 
                    #total travel time, roaming for bike from departure station to neighbour + cycling to arrival station

                    # calculate arrival time

                    # create an arrival event for the roaming user from the new departure station
                    world.add_event(
                        sim.BikeArrival(
                            self.time,
                            travel_time,
                            bike,
                            arrival_station.id,
                            closest_neighbour_with_bikes.id,
                        )
                    )

                    # remove bike from the new departure station
                    closest_neighbour_with_bikes.remove_bike(bike)

                    world.state.bike_in_use(bike)

                    departure_station.metrics.add_aggregate_metric(world, "roaming for bikes", 1)
                    world.metrics.add_aggregate_metric(world, "roaming for bikes", 1)
                    departure_station.metrics.add_aggregate_metric(world, "roaming distance for bikes", distance)
                    world.metrics.add_aggregate_metric(world, "roaming distance for bikes", distance)

                else:
                    departure_station.metrics.add_aggregate_metric(world, "starvation", 1) 
                    world.metrics.add_aggregate_metric(world, "starvation", 1)
                    rng_not_in_use = world.state.rng.choice(world.state.locations, p = p_normalized)

        departure_station.metrics.add_aggregate_metric(world, "trips", 1)
        world.metrics.add_aggregate_metric(world, "trips", 1)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, departing from station {self.departure_station_id}>"
    
    def acceptance_rejection(self,distance):
        prob_acceptance = -1.6548*distance**2-0.7036*distance+1.0133
        random_roaming_limit = random.triangular(0,MAX_ROAMING_DISTANCE)
        if random_roaming_limit <= prob_acceptance:
            return True
        else:
            return False 