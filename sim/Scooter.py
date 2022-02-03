from sim.Location import Location
from globals import BATTERY_LIMIT

class Scooter(Location):
    """
    E-scooter class containing state and all operations necessary
    """

    def __init__(self, lat: float = 0, lon: float = 0, battery: float = 100.0, scooter_id: int = 0):
        super().__init__(lat, lon, scooter_id)
        self.battery = battery
        self.battery_change_per_kilometer = 5.0

    def __deepcopy__(self, *args):
        return Scooter(self.lat, self.lon, self.battery, self.id)

    def travel(self, distance):
        self.battery -= distance * self.battery_change_per_kilometer

    def usable(self):
      return self.battery >= BATTERY_LIMIT

    def hasBattery(self):
      return True

    def swap_battery(self):
        self.battery = 100.0

    def __repr__(self):
        return f"ID-{self.id} B-{round(self.battery,1)}"
