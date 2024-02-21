from sim import Event
import sim
from settings import *

class EScooterArrival(Event):
    """
    Event performed when a bike arrives at a station after a bike departure
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
        self.arrival_station_id = arrival_area_id
        self.departure_station_id = departure_area_id
        self.travel_time = travel_time
        self.congested = congested

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        super().perform(world)

        # get arrival station 
        arrival_area = world.state.get_location_by_id(self.arrival_station_id)

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

            # add bike to the arrived station (location is changed in add_bike method)
            if arrival_area.add_bike(self.escooter):
                if FULL_TRIP:
                    world.state.remove_used_bike(self.escooter)
            else:
                if FULL_TRIP:
                    # go to another station
                    next_location = world.state.get_neighbouring_stations(arrival_area, 1, not_full=True)[0]

                    travel_time = world.state.get_travel_time(
                        arrival_area.location_id,
                        next_location.location_id,
                    )

                    # create an arrival event for the departed bike
                    world.add_event(
                        sim.BikeArrival(
                            self.time,
                            travel_time,
                            self.escooter,
                            next_location.location_id,
                            arrival_area.location_id,
                            congested = True
                        )
                    )

                else:
                    world.state.set_bike_in_use(self.escooter)

                
                distance = arrival_area.distance_to(next_location.get_lat(), next_location.get_lon())
                if distance <= MAX_ROAMING_DISTANCE_SOLUTIONS:
                    arrival_area.metrics.add_aggregate_metric(world, "short_congestion", 1)
                    world.metrics.add_aggregate_metric(world, "short_congestion", 1)
                else:
                    arrival_area.metrics.add_aggregate_metric(world, "long_congestion", 1)
                    world.metrics.add_aggregate_metric(world, "long_congestion", 1)
                    arrival_area.metrics.add_aggregate_metric(world, "Failed events", 1)
                    world.metrics.add_aggregate_metric(world, "Failed events", 1)
                
                arrival_area.metrics.add_aggregate_metric(world, "roaming distance for locks", distance)
                world.metrics.add_aggregate_metric(world, "roaming distance for locks", distance)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at station {self.arrival_station_id}>"
 