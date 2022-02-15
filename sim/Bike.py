from sim.Location import Location


class Bike(Location):
    """
    Bike class containing state and all operations necessary
    """

    def __init__(self, lat: float = 0, lon: float = 0, scooter_id: int = 0):
        super().__init__(lat, lon, scooter_id)

    # def __deepcopy__(self, *args):
    #     return Bike(self.lat, self.lon, self.id)

    def travel(self, distance):
      pass

    def usable(self):
      return True

    def hasBattery(self):
      return False

    def __repr__(self):
        return f"ID-{self.id}"
