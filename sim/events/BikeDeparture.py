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

    def __init__(self, departure_time, departure_station_id):
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
                mp = departure_station.get_move_probabilities(world.state, world.day(), world.hour())
                p = list(mp.values())
                sum = 0.0
                for i in range(len(p)):
                    sum += p[i]
                p_normalized = []
                for i in range(len(p)):
                    if sum > 0:
                        p_normalized.append(p[i] * (1.0/sum)) # TODO, not sure if this is needed
                    else:
                        p_normalized.append(1/len(p))
                arrival_station_id = world.state.rng.choice(list(mp.keys()), p = p_normalized)
                arrival_station = world.state.get_location_by_id(arrival_station_id)

                travel_time = world.state.get_travel_time(
                    departure_station.location_id,
                    arrival_station.location_id,
                )
                travel_time = travel_time if travel_time < 60 else 60 #TODO

                # create an arrival event for the departed bike
                world.add_event(
                    sim.BikeArrival(
                        self.time,
                        travel_time,
                        bike,
                        arrival_station.location_id,
                        departure_station.location_id,
                    )
                )

            # remove bike from the departure station
            departure_station.remove_bike(bike)

            world.state.set_bike_in_use(bike)

            world.metrics.add_aggregate_metric(world, "events", 2) #successfull pickup and an arrival

        else:
            if FULL_TRIP:
                closest_neighbour_with_bikes = world.state.get_neighbouring_stations(departure_station,1,not_empty=True)[0]
                distance = departure_station.distance_to(closest_neighbour_with_bikes.get_lat(), closest_neighbour_with_bikes.get_lon())
                mp=departure_station.get_move_probabilities(world.state, world.day(), world.hour())
                p = list(mp.values())
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
                    
                    arrival_station_id = world.state.rng.choice(list(mp.keys()), p = p_normalized)
                    arrival_station = world.state.get_location_by_id(arrival_station_id)

                    travel_time = world.state.get_travel_time(
                        closest_neighbour_with_bikes.location_id,
                        arrival_station.location_id,) + world.state.get_travel_time(departure_station.location_id,
                        closest_neighbour_with_bikes.location_id)*(BIKE_SPEED/WALKING_SPEED) 
                    #total travel time, roaming for bike from departure station to neighbour + cycling to arrival station

                    travel_time = travel_time if travel_time < 60 else 60 #TODO

                    # create an arrival event for the roaming user from the new departure station
                    world.add_event(
                        sim.BikeArrival(
                            self.time,
                            travel_time,
                            bike,
                            arrival_station.location_id,
                            closest_neighbour_with_bikes.location_id,
                        )
                    )

                    # remove bike from the new departure station
                    closest_neighbour_with_bikes.remove_bike(bike)

                    world.state.set_bike_in_use(bike)

                    world.metrics.add_aggregate_metric(world, "events", 2) #one roaming and an arrival

                    departure_station.metrics.add_aggregate_metric(world, "roaming for bikes", 1)
                    world.metrics.add_aggregate_metric(world, "roaming for bikes", 1)
                    departure_station.metrics.add_aggregate_metric(world, "roaming distance for bikes", distance)
                    world.metrics.add_aggregate_metric(world, "roaming distance for bikes", distance)

                else:
                    if departure_station.number_of_bikes() <= 0:
                        departure_station.metrics.add_aggregate_metric(world, "starvations, no bikes", 1) 
                        world.metrics.add_aggregate_metric(world, "starvations, no bikes", 1)
                    else:
                        departure_station.metrics.add_aggregate_metric(world, "starvations, no battery", 1) 
                        world.metrics.add_aggregate_metric(world, "starvations, no battery", 1)

                    world.metrics.add_aggregate_metric(world, "events", 1) #only one starvation --> lost demand and no arrival
                    departure_station.metrics.add_aggregate_metric(world, "starvation", 1) 
                    world.metrics.add_aggregate_metric(world, "starvation", 1)
                    departure_station.metrics.add_aggregate_metric(world, "Failed events", 1) 
                    world.metrics.add_aggregate_metric(world, "Failed events", 1)
                    
        departure_station.metrics.add_aggregate_metric(world, "trips", 1)
        world.metrics.add_aggregate_metric(world, "trips", 1)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, departing from station {self.departure_station_id}>"
    
    def acceptance_rejection(self,distance):
        prob_acceptance = -1.6548*distance**2-0.7036*distance+1.0133
        random_roaming_limit = random.uniform(0,1)
        if random_roaming_limit <= prob_acceptance:
            return True
        else:
            return False 