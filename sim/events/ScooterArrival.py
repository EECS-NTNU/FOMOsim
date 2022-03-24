from sim import Event
import sim
from settings import *

class ScooterArrival(Event):
    """
    Event performed when an e-scooter arrives at a cluster after a e-scooter departure
    """

    def __init__(
        self,
        arrival_time: int,
        scooter: sim.Scooter,
        arrival_cluster_id: int,
        departure_cluster_id: int,
        distance: int,
    ):
        super().__init__(arrival_time)
        self.scooter = scooter
        self.arrival_cluster_id = arrival_cluster_id
        self.departure_cluster_id = departure_cluster_id
        self.distance = distance

    def perform(self, world, **kwargs) -> None:
        """
        :param world: world object
        """

        # get arrival cluster
        arrival_cluster = world.state.get_location_by_id(self.arrival_cluster_id)

        if not FULL_TRIP:
            self.scooter = world.state.get_used_scooter()

        if self.scooter is not None:
            self.scooter.travel(self.distance)

            # add scooter to the arrived cluster (location is changed in add_scooter method)
            arrival_cluster.add_scooter(world.state.rng, self.scooter)

            if FULL_TRIP:
                world.state.remove_used_scooter(self.scooter)

        # set time of world to this event's time
        super(ScooterArrival, self).perform(world, **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at cluster {self.arrival_cluster_id}>"
