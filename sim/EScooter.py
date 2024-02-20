import sim
from sim.Bike import Bike
from settings import *
from sim.Location import Location
from sim.Metric import Metric


class EScooter(Location):
    """
    E-bike class containing state and all operations necessary
    """
    def __init__(self, lat: float = 0, lon: float = 0, location_id = 0, bike_id = 0, battery: float = 100.0):
      super().__init__(lat, lon, location_id)
      self.battery = battery
      self.metrics = Metric()
      self.bike_id = bike_id

    def travel(self, simul, travel_time, congested = False):
      if congested:
          self.metrics.add_metric(simul, "travel_time_congested", travel_time)
      else:
          self.metrics.add_metric(simul, "travel_time", travel_time)

    def __init__(self, lat = 0, lon = 0, battery = 100.0, bike_id = 0):
        super().__init__(lat, lon, bike_id)
        self.battery = battery
        self.battery_change_per_minute = BATTERY_CHANGE_PER_MINUTE

    def travel(self, simul, travel_time, congested = False):
        super().travel(simul, travel_time, congested)
        self.battery -= travel_time * self.battery_change_per_minute

    def usable(self):
      return self.battery >= BATTERY_LIMIT_TO_USE

    def hasBattery(self):
      return True

    def swap_battery(self):
       self.battery = 100.0

    def __repr__(self):
      return f"ID-{self.bike_id} B-{round(self.battery,1)}"
