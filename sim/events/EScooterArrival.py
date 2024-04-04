from sim import Event
import sim
from settings import *

class EScooterArrival(Event):
    """
    Event performed when a bike arrives at an area after a escooter departure
    """

    def __init__(
        self,
        time, 
        travel_time,
        escooter,
        arrival_area_id,
        departure_area_id,
        congested = False
    ):
        super().__init__(time + travel_time)
        self.escooter = escooter
        self.arrival_area_id = arrival_area_id
        self.departure_area_id = departure_area_id
        self.travel_time = travel_time
        self.congested = congested

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        super().perform(world)

        # get arrival area 
        arrival_area = world.state.get_location_by_id(self.arrival_area_id)

        if not FULL_TRIP:
            self.escooter = world.state.get_used_bike()

        if self.escooter is not None:
            self.escooter.travel(world, self.travel_time, self.congested)

            if self.escooter.battery < 0:
                arrival_area.metrics.add_aggregate_metric(world, "battery violation", 1)
                world.metrics.add_aggregate_metric(world, "battery violation", 1)
                arrival_area.metrics.add_aggregate_metric(world, "Failed events", 1)
                world.metrics.add_aggregate_metric(world, "Failed events", 1)
                self.escooter.battery = 0

            if FULL_TRIP:
                try:
                    world.state.remove_used_bike(self.escooter)
                except:
                    departure_area = world.state.get_location_by_id(self.departure_area_id)
                    bike_loc = world.state.get_location_by_id(self.escooter.location_id)
                    print("nå skjer det noe galt", self.escooter.location_id, self.arrival_area_id, self.departure_area_id)
                    print(world.state.get_location_by_id(self.escooter.location_id).bikes)

            arrival_area.add_bike(self.escooter)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at station {self.arrival_area_id}>"
 