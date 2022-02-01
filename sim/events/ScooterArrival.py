from sim import Event
import sim


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

        # make the scooter travel the distance to change battery
        self.scooter.travel(self.distance)

        # add scooter to the arrived cluster (location is changed in add_scooter method)
        arrival_cluster.add_scooter(self.scooter)

        # adding the trip to world flow for visualizing purposes
        world.add_trip_to_flow(self.departure_cluster_id, self.arrival_cluster_id)

        # set time of world to this event's time
        super(ScooterArrival, self).perform(world, **kwargs)
