from sim.Location import Location
from settings import *

class Bike(Location):
    """
    Bike class containing state and all operations necessary
    """

    def __init__(self, lat: float = 0, lon: float = 0, scooter_id: int = 0):
        super().__init__(lat, lon, scooter_id)
        self.battery = 100.0

    def travel(self, distance):
      pass

    def usable(self):
      return True

    def hasBattery(self):
      return False

    def __repr__(self):
        return f"ID-{self.id}"
