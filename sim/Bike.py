from sim.Location import Location
from sim.Metric import Metric
from settings import *

class Bike(Location):
    """
    Bike class containing state and all operations necessary
    """

    def __init__(self, lat: float = 0, lon: float = 0, bike_id: int = 0):
        super().__init__(lat, lon, bike_id)
        self.battery = 100.0
        self.metrics = Metric()

    def travel(self, simul, travel_time, congested = False):
        # print("normal travel")
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
