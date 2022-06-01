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
            if arrival_cluster.add_scooter(world.state.rng, self.scooter):
                if FULL_TRIP:
                    world.state.remove_used_scooter(self.scooter)
            else:
                if FULL_TRIP:
                    # go to another station
                    neighbours = world.state.get_neighbours(arrival_cluster, 2, exclude=[depot.id for depot in world.state.depots])
                    next_cluster = world.state.rng.choice(neighbours)

                    trip_distance = world.state.get_distance(
                        arrival_cluster.id,
                        next_cluster.id,
                    )

                    trip_speed = world.state.get_trip_speed(
                        arrival_cluster.id,
                        next_cluster.id,
                    )

                    # calculate arrival time
                    if trip_speed == 0.0:
                        pass # debug issue10
                        arrival_time = self.time + 10 # debug
                        loggWrite("arrival_time set by debug code")
                    else:
                        arrival_time = self.time + round((trip_distance / trip_speed) * 60)

                    # create an arrival event for the departed scooter
                    world.add_event(
                        sim.ScooterArrival(
                            arrival_time,
                            self.scooter,
                            next_cluster.id,
                            arrival_cluster.id,
                            trip_distance,
                        )
                    )

                else:
                    world.state.scooter_in_use(self.scooter)

                world.metrics.add_aggregate_metric(world, "congestion", 1)

        # set time of world to this event's time
        super(ScooterArrival, self).perform(world, **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at cluster {self.arrival_cluster_id}>"
