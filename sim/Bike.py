from sim.Location import Location
from sim.Metric import Metric
from settings import *

class Bike(Location):
    """
    Bike class containing state and all operations necessary
    """

    def __init__(self,
                 lat: float = 0, 
                 lon: float = 0, 
                 location_id = 0, 
                 bike_id = 0):
        super().__init__(lat, lon, location_id)
        self.is_station_based = True
        self.metrics = Metric()
        self.bike_id = bike_id
        self.battery = 100.0

    def travel(self, simul, travel_time, congested = False):
        if congested:
            self.metrics.add_metric(simul, "travel_time_congested", travel_time)
        else:
            self.metrics.add_metric(simul, "travel_time", travel_time)

    def usable(self):
      return True

    def hasBattery(self):
      return False

    def __repr__(self):
        return f"ID-{self.id}-{self.lat}-{self.lon}"
