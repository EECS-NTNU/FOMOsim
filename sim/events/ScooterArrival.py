from sim import Event
import sim
from settings import *

class ScooterArrival(Event):
    """
    Event performed when an e-scooter arrives at a station after a e-scooter departure
    """

    def __init__(
        self,
        time, 
        travel_time,
        scooter: sim.Scooter,
        arrival_station_id: int,
        departure_station_id: int,
    ):
        super().__init__(time + travel_time)
        self.scooter = scooter
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
            self.scooter = world.state.get_used_scooter()

        if self.scooter is not None:
            self.scooter.travel(self.travel_time)

            # add scooter to the arrived station (location is changed in add_scooter method)
            if arrival_station.add_scooter(world.state.rng, self.scooter):
                if FULL_TRIP:
                    world.state.remove_used_scooter(self.scooter)
            else:
                if FULL_TRIP:
                    # go to another station
                    next_station = world.state.get_neighbours(arrival_station, 1, not_full=True)[0]

                    travel_time = world.state.get_travel_time(
                        arrival_station.id,
                        next_station.id,
                    )

                    # create an arrival event for the departed scooter
                    world.add_event(
                        sim.ScooterArrival(
                            self.time,
                            travel_time,
                            self.scooter,
                            next_station.id,
                            arrival_station.id,
                        )
                    )

                else:
                    world.state.scooter_in_use(self.scooter)

                world.metrics.add_aggregate_metric(world, "congestion", 1)

        # set time of world to this event's time
        super(ScooterArrival, self).perform(world)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at station {self.arrival_station_id}>"
