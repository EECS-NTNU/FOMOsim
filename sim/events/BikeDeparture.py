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

    def perform(self, simul) -> None:
        """
        :param simul: Simulation object
        """

        super().perform(simul)

        # get departure station
        departure_station = simul.state.get_location_by_id(self.departure_station_id)

        # get all available bike in the station
        available_bikes = departure_station.get_available_bikes()

        # if there are no more available bikes -> make a LostTrip event for that departure time
        if len(available_bikes) > 0:
            bike = available_bikes.pop(0)

            if FULL_TRIP:
                if simul.state.rng.random() < RANDOM_DESTINATION_PROB:
                    # Exclude the current area from the random selection
                    other_stations = [station.id for station in simul.state.get_stations() if station.id != self.departure_station_id]
                    arrival_station_id = simul.state.rng.choice(other_stations)
                else:
                    # get an arrival station from the leave prob distribution
                    mp = departure_station.get_move_probabilities(simul.state, simul.state.day(), simul.state.hour())
                    p = list(mp.values())
                    sum = 0.0
                    for i in range(len(p)):
                        sum += p[i]
                    p_normalized = []
                    for i in range(len(p)):
                        if sum > 0:
                            p_normalized.append(p[i] * (1.0/sum))
                        else:
                            p_normalized.append(1/len(p))
                    arrival_station_id = simul.state.rng.choice(list(mp.keys()), p = p_normalized)

                arrival_station = simul.state.get_location_by_id(arrival_station_id)

                travel_time = simul.state.get_travel_time(
                    departure_station.id,
                    arrival_station.id,
                )


                # create an arrival event for the departed bike
                simul.add_event(
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

            simul.state.set_bike_in_use(bike)

            simul.state.metrics.add_aggregate_metric(simul.state, "bike departure", 1)
            simul.state.metrics.add_aggregate_metric(simul.state, "events", 2)

        else:
            if FULL_TRIP:
                closest_neighbour_with_bikes = simul.state.get_neighbouring_stations(departure_station,1,not_empty=True)[0]
                distance = departure_station.distance_to(closest_neighbour_with_bikes.get_lat(), closest_neighbour_with_bikes.get_lon())
                mp=departure_station.get_move_probabilities(simul.state, simul.state.day(), simul.state.hour())
                p = list(mp.values())
                sum = 0.0
                for i in range(len(p)):
                    sum += p[i]
                p_normalized = []
                for i in range(len(p)):
                    if sum > 0:
                        p_normalized.append(p[i] * (1.0/sum))
                    else:
                        p_normalized.append(1/len(p))
                if self.acceptance_rejection(distance, simul):
                    available_bikes = closest_neighbour_with_bikes.get_available_bikes()
                    bike=available_bikes.pop(0)
                    
                    if simul.state.rng.random() < RANDOM_DESTINATION_PROB:
                        # Exclude the current area from the random selection
                        other_stations = [station.id for station in simul.state.get_stations() if station.id != self.departure_station_id]
                        arrival_station_id = simul.state.rng.choice(other_stations)
                    else:
                        arrival_station_id = simul.state.rng.choice(list(mp.keys()), p = p_normalized)
                    
                    arrival_station = simul.state.get_location_by_id(arrival_station_id)

                    travel_time = simul.state.get_travel_time(
                        closest_neighbour_with_bikes.id,
                        arrival_station.id,) + simul.state.get_travel_time(departure_station.id,
                        closest_neighbour_with_bikes.id)*(BIKE_SPEED/WALKING_SPEED) 
                    #total travel time, roaming for bike from departure station to neighbour + cycling to arrival station

                    # calculate arrival time 

                    # create an arrival event for the roaming user from the new departure station
                    simul.add_event(
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

                    simul.state.set_bike_in_use(bike)

                    simul.state.metrics.add_aggregate_metric(simul.state, "bike departure", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "events", 2)

                    simul.state.metrics.add_aggregate_metric(simul.state, "roaming for bikes", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "roaming distance for bikes", distance)

                else:
                    if departure_station.number_of_bikes() <= 0:
                        simul.state.metrics.add_aggregate_metric(simul.state, "bike starvations", 1)
                    else:
                        simul.state.metrics.add_aggregate_metric(simul.state, "battery starvations", 1)

                    simul.state.metrics.add_aggregate_metric(simul.state, "events", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "starvations", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "failed events", 1)
                    
        simul.state.metrics.add_aggregate_metric(simul.state, "trips", 1)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, departing from station {self.departure_station_id}>"
    
    def acceptance_rejection(self,distance, simul):
        prob_acceptance = -1.6548*distance**2-0.7036*distance+1.0133
        random_roaming_limit = simul.state.rng.uniform(0,1)
        if random_roaming_limit <= prob_acceptance:
            return True
        else:
            return False 