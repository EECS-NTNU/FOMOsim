import sim
from sim import Event
from settings import *
import numpy as np

class BikeDeparture(Event):
    """
    Event fired when a customer requests a trip from a given departure station. Creates a Lost Trip or Bike Arrival
    event based on the availability of the station
    """

    def __init__(self, departure_time: int, departure_station_id: int):
        super().__init__(departure_time)
        self.departure_station_id = departure_station_id

    def perform(self, simul) -> None:
        """
        :param simul: simul object
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
                # get a arrival station from the leave prob distribution

                p=departure_station.get_move_probabilities(simul.state, simul.state.day(), simul.state.hour())
                sum = 0.0
                for i in range(len(p)):
                    sum += p[i]
                p_normalized = []
                for i in range(len(p)):
                    if sum > 0:
                        p_normalized.append(p[i] * (1.0/sum)) # TODO, not sure if this is needed
                    else:
                        p_normalized.append(1/len(p))
                arrival_station = simul.state.rng.choice(simul.state.locations, p = p_normalized)

                travel_time = simul.state.get_travel_time(
                    departure_station.id,
                    arrival_station.id,
                )

                # calculate arrival time

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

            simul.state.bike_in_use(bike)

        else:
            departure_station.metrics.add_aggregate_metric(simul, "starvation", 1)
            simul.metrics.add_aggregate_metric(simul, "starvation", 1)

        departure_station.metrics.add_aggregate_metric(simul, "trips", 1)
        simul.metrics.add_aggregate_metric(simul, "trips", 1)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, departing from station {self.departure_station_id}>"
