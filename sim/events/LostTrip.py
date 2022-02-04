from sim import Event


class LostTrip(Event):
    """
    Event for a lost trip recording a lost trip for the world object
    """

    def __init__(self, time: int, location_id: int):
        super().__init__(time)
        self.location_id = location_id

    def perform(self, world, **kwargs) -> None:
        world.metrics.add_metric(world, "lost_demand", 1)
#        if world.verbose:
#            print(f"LT: {self.location_id} at {self.time}")
        super(LostTrip, self).perform(world, **kwargs)
