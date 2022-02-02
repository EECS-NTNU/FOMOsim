import sim
from sim import Event
from globals import SCOOTER_SPEED
import numpy as np


class ScooterDeparture(Event):
    """
    Event fired when a customer requests a trip from a given departure cluster. Creates a Lost Trip or Scooter arrival
    event based on the availability of the cluster
    """

    def __init__(self, departure_time: int, departure_cluster_id: int):
        super().__init__(departure_time)
        self.departure_cluster_id = departure_cluster_id

    def perform(self, world, **kwargs) -> None:
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

            # get a arrival cluster from the leave prob distribution
            arrival_cluster = np.random.choice(
                world.state.stations, p=departure_cluster.get_leave_distribution()
            )

            trip_distance = world.state.get_distance(
                departure_cluster.id,
                arrival_cluster.id,
            )

            # calculate arrival time
            arrival_time = self.time + round(trip_distance / SCOOTER_SPEED * 60)

            # create an arrival event for the departed scooter
            world.add_event(
                sim.ScooterArrival(
                    arrival_time,
                    scooter,
                    arrival_cluster.id,
                    departure_cluster.id,
                    trip_distance,
                )
            )

            # remove scooter from the departure cluster
            departure_cluster.remove_scooter(scooter)
        else:
            world.add_event(sim.LostTrip(self.time, self.departure_cluster_id))

        # set time of world to this event's time
        super(ScooterDeparture, self).perform(world, **kwargs)
