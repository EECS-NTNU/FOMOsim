from sim import Event
import sim
from settings import *

class ScooterArrival(Event):
    """
    Event performed when an e-scooter arrives at a cluster after a e-scooter departure
    """

    def __init__(
        self,
        time, 
        travel_time,
        scooter: sim.Scooter,
        arrival_cluster_id: int,
        departure_cluster_id: int,
    ):
        super().__init__(time + travel_time)
        self.scooter = scooter
        self.arrival_cluster_id = arrival_cluster_id
        self.departure_cluster_id = departure_cluster_id
        self.travel_time = travel_time

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        # get arrival cluster
        arrival_cluster = world.state.get_location_by_id(self.arrival_cluster_id)

        if not FULL_TRIP:
            self.scooter = world.state.get_used_scooter()

        if self.scooter is not None:
            self.scooter.travel(self.travel_time)

            # add scooter to the arrived cluster (location is changed in add_scooter method)
            if arrival_cluster.add_scooter(world.state.rng, self.scooter):
                if FULL_TRIP:
                    world.state.remove_used_scooter(self.scooter)
            else:
                if FULL_TRIP:
                    # go to another station
                    next_cluster = world.state.get_neighbours(arrival_cluster, 1, not_full=True, exclude=[depot.id for depot in world.state.depots])[0]

                    travel_time = world.state.get_travel_time(
                        arrival_cluster.id,
                        next_cluster.id,
                    )

                    # create an arrival event for the departed scooter
                    world.add_event(
                        sim.ScooterArrival(
                            self.time,
                            travel_time,
                            self.scooter,
                            next_cluster.id,
                            arrival_cluster.id,
                        )
                    )

                else:
                    world.state.scooter_in_use(self.scooter)

                world.metrics.add_aggregate_metric(world, "congestion", 1)

        # set time of world to this event's time
        super(ScooterArrival, self).perform(world)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at cluster {self.arrival_cluster_id}>"
