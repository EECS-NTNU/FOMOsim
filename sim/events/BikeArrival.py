from sim import Event
import sim
from settings import *

class BikeArrival(Event):
    """
    Event performed when a bike arrives at a station after a bike departure
    """

    def __init__(
        self,
        time, 
        travel_time,
        bike,
        arrival_station_id,
        departure_station_id,
        congested = False
    ):
        super().__init__(time + travel_time)
        self.bike = bike
        self.arrival_station_id = arrival_station_id
        self.departure_station_id = departure_station_id
        self.travel_time = travel_time
        self.congested = congested

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        super().perform(world)

        # get arrival station 
        arrival_station = world.state.get_location_by_id(self.arrival_station_id)

        if not FULL_TRIP:
            self.bike = world.state.get_used_bike()

        if self.bike is not None:
            self.bike.travel(world, self.travel_time, self.congested)

            if self.bike.battery < 0:
                arrival_station.metrics.add_aggregate_metric(world, "battery violation", 1)
                world.metrics.add_aggregate_metric(world, "battery violation", 1)
                arrival_station.metrics.add_aggregate_metric(world, "Failed events", 1)
                world.metrics.add_aggregate_metric(world, "Failed events", 1)
                self.bike.battery = 0

            # add bike to the arrived station (location is changed in add_bike method)
            if arrival_station.add_bike(self.bike):
                if FULL_TRIP:
                    world.state.remove_used_bike(self.bike)
            else:
                if FULL_TRIP:
                    # go to another station
                    next_station = world.state.get_neighbouring_stations(arrival_station, 1, not_full=True)[0]

                    travel_time = world.state.get_travel_time(
                        arrival_station.location_id,
                        next_station.location_id,
                    )

                    # create an arrival event for the departed bike
                    world.add_event(
                        sim.BikeArrival(
                            self.time,
                            travel_time,
                            self.bike,
                            next_station.location_id,
                            arrival_station.location_id,
                            congested = True
                        )
                    )

                else:
                    world.state.set_bike_in_use(self.bike)

                
                distance = arrival_station.distance_to(next_station.get_lat(), next_station.get_lon())
                if distance <= MAX_ROAMING_DISTANCE_SOLUTIONS:
                    arrival_station.metrics.add_aggregate_metric(world, "short_congestion", 1)
                    world.metrics.add_aggregate_metric(world, "short_congestion", 1)
                else:
                    arrival_station.metrics.add_aggregate_metric(world, "long_congestion", 1)
                    world.metrics.add_aggregate_metric(world, "long_congestion", 1)
                    arrival_station.metrics.add_aggregate_metric(world, "Failed events", 1)
                    world.metrics.add_aggregate_metric(world, "Failed events", 1)
                
                arrival_station.metrics.add_aggregate_metric(world, "roaming distance for locks", distance)
                world.metrics.add_aggregate_metric(world, "roaming distance for locks", distance)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at station {self.arrival_station_id}>"
 