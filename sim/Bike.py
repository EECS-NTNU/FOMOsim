from sim.Location import Location
from settings import *

class Bike(Location):
    """
    Bike class containing state and all operations necessary
    """

    def __init__(self, lat: float = 0, lon: float = 0, bike_id: int = 0):
        super().__init__(lat, lon, bike_id)
        self.battery = 100.0

    def travel(self, travel_time):
      pass

    def usable(self):
      return True

    def hasBattery(self):
      return False

    def __repr__(self):
        return f"ID-{self.id}-{self.lat}-{self.lon}"
