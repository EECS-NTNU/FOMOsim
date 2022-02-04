import sim
from sim import Event
from settings import *
import numpy as np


class GenerateScooterTrips(Event):
    """
    This event creates e-scooter departure events based on the trip intensity parameter and a Possion Distribution
    """

    def __init__(self, time: int):
        super().__init__(time)

    def perform(self, world, **kwargs) -> None:
        for departure_cluster in world.state.stations:
            # poisson process to select number of trips in a iteration
            number_of_trips = round(
                np.random.poisson(departure_cluster.leave_intensity_per_iteration)
            )

            # generate trip departure times (can be implemented with np.random.uniform if we want decimal times)
            # both functions generate numbers from a discrete uniform distribution
            trips_departure_time = sorted(
                np.random.randint(
                    self.time, self.time + ITERATION_LENGTH_MINUTES, number_of_trips
                )
            )

            # generate departure event and add to world stack
            for departure_time in trips_departure_time:
                # add departure event to the stack
                departure_event = sim.ScooterDeparture(
                    departure_time, departure_cluster.id
                )
                world.add_event(departure_event)

        if not FULL_TRIP:
            for arrival_cluster in world.state.stations:
                # poisson process to select number of trips in a iteration
                number_of_trips = round(
                    np.random.poisson(arrival_cluster.arrive_intensity_per_iteration)
                )

                # generate trip arrival times (can be implemented with np.random.uniform if we want decimal times)
                # both functions generate numbers from a discrete uniform distribution
                trips_arrival_time = sorted(
                    np.random.randint(
                        self.time, self.time + ITERATION_LENGTH_MINUTES, number_of_trips
                    )
                )

                # generate arrival event and add to world stack
                for arrival_time in trips_arrival_time:
                    # add arrival event to the stack
                    arrival_event = sim.ScooterArrival(
                        arrival_time,
                        None,
                        arrival_cluster.id,
                        None,
                        0,
                    )
                    world.add_event(arrival_event)

        world.add_event(GenerateScooterTrips(self.time + ITERATION_LENGTH_MINUTES))

        super(GenerateScooterTrips, self).perform(world, add_metric=False)
