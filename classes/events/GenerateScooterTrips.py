import classes
from classes import Event
from globals import ITERATION_LENGTH_MINUTES
import numpy as np


class GenerateScooterTrips(Event):
    """
    This event creates e-scooter departure events based on the trip intensity parameter and a Possion Distribution
    """

    def __init__(self, time: int):
        super().__init__(time)

    def perform(self, world, **kwargs) -> None:
        for departure_cluster in world.state.clusters:
            # poisson process to select number of trips in a iteration
            number_of_trips = round(
                np.random.poisson(departure_cluster.trip_intensity_per_iteration)
            )

            # generate trip departure times (can be implemented with np.random.uniform if we want decimal times)
            # both functions generate numbers from a discrete uniform distribution
            trips_departure_time = sorted(
                np.random.randint(
                    self.time, self.time + ITERATION_LENGTH_MINUTES, number_of_trips
                )
            )

            # generate departure and arrival event for every trip and add to world stack
            for departure_time in trips_departure_time:
                # add departure event to the stack
                departure_event = classes.ScooterDeparture(
                    departure_time, departure_cluster.id
                )
                world.add_event(departure_event)

        world.add_event(GenerateScooterTrips(self.time + ITERATION_LENGTH_MINUTES))

        super(GenerateScooterTrips, self).perform(world, add_metric=False)
