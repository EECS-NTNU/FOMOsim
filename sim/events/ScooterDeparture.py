import sim
from sim import Event
from settings import *
import numpy as np

from helpers import loggWrite

class ScooterDeparture(Event):
    """
    Event fired when a customer requests a trip from a given departure cluster. Creates a Lost Trip or Scooter arrival
    event based on the availability of the cluster
    """

    def __init__(self, departure_time: int, departure_cluster_id: int):
        super().__init__(departure_time)
        self.departure_cluster_id = departure_cluster_id

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        # get departure cluster
        departure_cluster = world.state.get_location_by_id(self.departure_cluster_id)

        # get all available scooter in the cluster
        available_scooters = departure_cluster.get_available_scooters()

        # if there are no more available scooters -> make a LostTrip event for that departure time
        if len(available_scooters) > 0:
            scooter = available_scooters.pop(0)

            if FULL_TRIP:
                # get a arrival cluster from the leave prob distribution

                p=departure_cluster.get_leave_distribution(world.state, world.day(), world.hour())
                sum = 0.0
                for i in range(len(p)):
                    sum += p[i]
                p_normalized = []
                for i in range(len(p)):
                    if sum > 0:
                        p_normalized.append(p[i] * (1.0/sum)) # TODO, not sure if this is needed
                    else:
                        p_normalized.append(1/len(p))
                arrival_cluster = world.state.rng.choice(world.state.locations, p = p_normalized)

                travel_time = world.state.get_travel_time(
                    departure_cluster.id,
                    arrival_cluster.id,
                )

                # calculate arrival time

                # create an arrival event for the departed scooter
                world.add_event(
                    sim.ScooterArrival(
                        self.time,
                        travel_time,
                        scooter,
                        arrival_cluster.id,
                        departure_cluster.id,
                    )
                )

            # remove scooter from the departure cluster
            departure_cluster.remove_scooter(scooter)

            world.state.scooter_in_use(scooter)

        else:
            world.metrics.add_aggregate_metric(world, "lost_demand", 1)

        world.metrics.add_aggregate_metric(world, "trips", 1)

        # set time of world to this event's time
        super(ScooterDeparture, self).perform(world)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, departing from cluster {self.departure_cluster_id}>"
