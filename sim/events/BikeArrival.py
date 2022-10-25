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
        arrival_station_id: int,
        departure_station_id: int,
    ):
        super().__init__(time + travel_time)
        self.bike = bike
        self.arrival_station_id = arrival_station_id
        self.departure_station_id = departure_station_id
        self.travel_time = travel_time

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        # get arrival station
        arrival_station = world.state.get_location_by_id(self.arrival_station_id)

        if not FULL_TRIP:
            self.bike = world.state.get_used_bike()

        if self.bike is not None:
            self.bike.travel(self.travel_time)

            # add bike to the arrived station (location is changed in add_bike method)
            if arrival_station.add_bike(self.bike):
                if FULL_TRIP:
                    world.state.remove_used_bike(self.bike)
            else:
                if FULL_TRIP:
                    # go to another station
                    next_station = world.state.get_neighbours(arrival_station, 1, not_full=True)[0]

                    travel_time = world.state.get_travel_time(
                        arrival_station.id,
                        next_station.id,
                    )

                    # create an arrival event for the departed bike
                    world.add_event(
                        sim.BikeArrival(
                            self.time,
                            travel_time,
                            self.bike,
                            next_station.id,
                            arrival_station.id,
                        )
                    )

                else:
                    world.state.bike_in_use(self.bike)

                arrival_station.metrics.add_aggregate_metric(world, "congestion", 1)
                world.metrics.add_aggregate_metric(world, "congestion", 1)

        # set time of world to this event's time
        super(BikeArrival, self).perform(world)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at station {self.arrival_station_id}>"
