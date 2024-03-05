import numpy as np
import sim
from sim import Event
import settings
from settings import *
from helpers import loggDepartures

class GenerateBikeTrips(Event):
    """
    This event creates bike departure events based on the trip intensity parameter and a Possion Distribution
    """

    def __init__(self, time: int):
        super().__init__(time)

    def perform(self, simul) -> None:

        super().perform(simul)

        for departure_station in simul.state.locations:
            # poisson process to select number of trips in a iteration
            number_of_trips = round(
                simul.state.rng.poisson(departure_station.get_leave_intensity(simul.state.day(), simul.state.hour()) / (60/ITERATION_LENGTH_MINUTES))
            )

            # generate trip departure times (can be implemented with np.random.uniform if we want decimal times)
            # both functions generate numbers from a discrete uniform distribution

            trips_departure_time = sorted(
                simul.state.rng.integers(
                    self.time, self.time + ITERATION_LENGTH_MINUTES, number_of_trips
                )
            )
            if settings.TRAFFIC_LOGGING and len(trips_departure_time) > 0:
                loggDepartures(departure_station.id, trips_departure_time) 

            # generate departure event and add to simul event_queue
            for departure_time in trips_departure_time:
                # add departure event to the event_queue
                departure_event = sim.BikeDeparture(
                    departure_time, departure_station.id
                )
                simul.add_event(departure_event)

        if not FULL_TRIP:
            for arrival_station in simul.state.locations:
                # poisson process to select number of trips in a iteration
                number_of_trips = round(
                    simul.state.rng.poisson(arrival_station.get_arrive_intensity(simul.state.day(), simul.state.hour()) / (60/ITERATION_LENGTH_MINUTES))
                )

                # generate trip arrival times (can be implemented with np.random.uniform if we want decimal times)
                # both functions generate numbers from a discrete uniform distribution
                trips_arrival_time = sorted(
                    simul.state.rng.integers(
                        self.time, self.time + ITERATION_LENGTH_MINUTES, number_of_trips
                    )
                )

                # generate arrival event and add to simul event_queue
                for arrival_time in trips_arrival_time:
                    # add arrival event to the event_queue
                    arrival_event = sim.BikeArrival(
                        arrival_time,
                        None,
                        arrival_station.id,
                        None,
                        0,
                    )
                    simul.add_event(arrival_event)

        simul.add_event(GenerateBikeTrips(self.time + ITERATION_LENGTH_MINUTES))
