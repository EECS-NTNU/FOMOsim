import numpy as np
import sim
from sim import Event
import settings
from settings import *
from helpers import loggDepartures
import csv

class GenerateBikeTrips(Event):
    """
    This event creates bike departure events based on the trip intensity parameter and a Possion Distribution
    """

    def __init__(self, time: int):
        super().__init__(time)

    def perform(self, world) -> None:

        super().perform(world)

        for departure_station in world.state.locations:
            # poisson process to select number of trips in a iteration
            number_of_trips = round(
                world.state.rng.poisson(departure_station.get_leave_intensity(world.day(), world.hour()) / (60/ITERATION_LENGTH_MINUTES))
            )

            # generate trip departure times (can be implemented with np.random.uniform if we want decimal times)
            # both functions generate numbers from a discrete uniform distribution

            trips_departure_time = sorted(
                world.state.rng.integers(
                    self.time, self.time + ITERATION_LENGTH_MINUTES, number_of_trips
                )
            )
            if settings.TRAFFIC_LOGGING and len(trips_departure_time) > 0:
                loggDepartures(departure_station.id, trips_departure_time) 

            # generate departure event and add to world event_queue
            for departure_time in trips_departure_time:
                # add departure event to the event_queue
                departure_event = sim.BikeDeparture(
                    departure_time, departure_station.id
                )
                world.add_event(departure_event)
                path= 'policies/inngjerdingen_moeller/simulation_results/innMoll.csv'
                with open(path,'a', newline='') as f:
                    writer=csv.writer(f)
                    writer.writerow([departure_station.id,int(departure_time)])
                

        if not FULL_TRIP: 
            for arrival_station in world.state.locations:
                # poisson process to select number of trips in a iteration
                number_of_trips = round(
                    world.state.rng.poisson(arrival_station.get_arrive_intensity(world.day(), world.hour()) / (60/ITERATION_LENGTH_MINUTES))
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
                    arrival_event = sim.BikeArrival(
                        arrival_time,
                        None,
                        arrival_station.id,
                        None,
                        0,
                    )
                    world.add_event(arrival_event)

        world.add_event(GenerateBikeTrips(self.time + ITERATION_LENGTH_MINUTES))
