import sim
from settings import *
from sim.Location import Location
from sim.Metric import Metric


class EScooter(Location):
    """
    E-Scooter class
    """
    def __init__(self, 
                 lat: float, 
                 lon: float, 
                 location_id, 
                 bike_id, 
                 battery: float = 100.0):
      super().__init__(lat, lon, location_id)
      self.battery = battery
      self.metrics = Metric()
      self.bike_id = bike_id
      self.battery_change_per_minute = BATTERY_CHANGE_PER_MINUTE

    def travel(self, simul, travel_time, congested = False):
      if congested:
          self.metrics.add_metric(simul, "travel_time_congested", travel_time)
      else:
          self.metrics.add_metric(simul, "travel_time", travel_time)

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
