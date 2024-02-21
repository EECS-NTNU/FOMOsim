import numpy as np
import sim
from sim import Event
import settings
from settings import *
from helpers import loggDepartures
import csv

class GenerateEScooterTrips(Event):
    """
    This event creates bike departure events based on the trip intensity parameter and a Possion Distribution
    """

    def __init__(self, time: int):
        super().__init__(time)

    def perform(self, world) -> None:

        super().perform(world)

        for departure_area in world.state.get_areas():
            # poisson process to select number of trips in a iteration
            number_of_trips = 2*round(
                world.state.rng.poisson(departure_area.get_leave_intensity(world.day(), world.hour()) / (60/ITERATION_LENGTH_MINUTES))
            )

            # generate trip departure times (can be implemented with np.random.uniform if we want decimal times)
            # both functions generate numbers from a discrete uniform distribution
            trips_departure_time = sorted(
                world.state.rng.integers(
                    self.time, self.time + ITERATION_LENGTH_MINUTES, number_of_trips
                )
            )

            if settings.TRAFFIC_LOGGING and len(trips_departure_time) > 0:
                loggDepartures(departure_area.location_id, trips_departure_time) 

            # generate departure event and add to world event_queue
            for departure_time in trips_departure_time:
                # add departure event to the event_queue
                departure_event = sim.EScooterDeparture(
                    departure_time, departure_area.location_id
                )
                world.add_event(departure_event)
        
        if not FULL_TRIP: 
            for arrival_area in world.state.get_areas():
                # poisson process to select number of trips in a iteration
                number_of_trips = round(
                    world.state.rng.poisson(arrival_area.get_arrive_intensity(world.day(), world.hour()) / (60/ITERATION_LENGTH_MINUTES))
                )

                # generate trip arrival times (can be implemented with np.random.uniform if we want decimal times)
                # both functions generate numbers from a discrete uniform distribution
                trips_arrival_time = sorted(
                    world.state.rng.integers(
                        self.time, self.time + ITERATION_LENGTH_MINUTES, number_of_trips
                    )
                )

                # generate arrival event and add to world event_queue
                for arrival_time in trips_arrival_time:
                    # add arrival event to the event_queue
                    arrival_event = sim.EScooterArrival(
                        arrival_time,
                        None,
                        arrival_area.location_id,
                        None,
                        0,
                    )
                    world.add_event(arrival_event)

        world.add_event(GenerateEScooterTrips(self.time + ITERATION_LENGTH_MINUTES))
