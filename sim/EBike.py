import sim
from sim.Bike import Bike
from settings import *

class EBike(Bike):
    """
    E-bike class containing state and all operations necessary
    """

    def __init__(self, lat = 0, lon = 0, battery = 100.0, location_id = 0, bike_id = 0):
        super().__init__(lat, lon, location_id)
        self.battery = battery
        self.battery_change_per_minute = BATTERY_CHANGE_PER_MINUTE
        self.bike_id = bike_id

    def travel(self, simul, travel_time, congested = False):
        super().travel(simul, travel_time, congested)
        self.battery -= travel_time * self.battery_change_per_minute

    def usable(self):
      return self.battery >= BATTERY_LIMIT

    def hasBattery(self):
      return True

    def swap_battery(self):
        self.battery = 100.0

    def __repr__(self):
        return f"ID-{self.id} B-{round(self.battery,1)}"
