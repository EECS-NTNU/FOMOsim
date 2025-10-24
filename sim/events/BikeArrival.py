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

    def perform(self, simul) -> None:
        """
        :param simul: Simulation object
        """

        super().perform(simul)

        # get arrival station 
        arrival_station = simul.state.get_location_by_id(self.arrival_station_id)

        if not FULL_TRIP:
            self.bike = simul.state.get_used_bike()

        if self.bike is not None:
            self.bike.travel(simul, self.travel_time, self.congested)

            if self.bike.battery < 0:
                simul.state.metrics.add_aggregate_metric(simul.state, "battery violations", 1)
                simul.state.metrics.add_aggregate_metric(simul.state, "failed events", 1)
                self.bike.battery = 0

            # add bike to the arrived station (location is changed in add_bike method)
            if arrival_station.add_bike(self.bike):
                if FULL_TRIP:
                    simul.state.remove_used_bike(self.bike)
                
                simul.state.metrics.add_aggregate_metric(simul.state, "bike arrival", 1)

            else:
                if FULL_TRIP:
                    # go to another station
                    next_station = simul.state.get_neighbouring_stations(arrival_station, 1, not_full=True)[0]

                    travel_time = simul.state.get_travel_time(
                        arrival_station.id,
                        next_station.id,
                    )

                    # create an arrival event for the departed bike
                    simul.add_event(
                        sim.BikeArrival(
                            self.time,
                            travel_time,
                            self.bike,
                            next_station.id,
                            arrival_station.id,
                            congested = True
                        )
                    )

                    simul.state.metrics.add_aggregate_metric(simul.state, "events", 1)


                else:
                    simul.state.set_bike_in_use(self.bike)

                distance = arrival_station.distance_to(next_station.get_lat(), next_station.get_lon())
                if distance <= MAX_ROAMING_DISTANCE_SOLUTIONS:
                    simul.state.metrics.add_aggregate_metric(simul.state, "short congestions", 1)
                else:
                    simul.state.metrics.add_aggregate_metric(simul.state, "long congestions", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "failed events", 1)
                
                simul.state.metrics.add_aggregate_metric(simul.state, "roaming for locks", 1)
                simul.state.metrics.add_aggregate_metric(simul.state, "roaming distance for locks", distance)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at station {self.arrival_station_id}>"
 