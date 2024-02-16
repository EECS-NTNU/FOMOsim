import geopy.distance
from settings import *

class Location:
    """
    Base location class. All classes representing a geographic position inherit from the Location class
    """

    def __init__(self, lat, lon, location_id):
        self.lat = lat
        self.lon = lon
        self.location_id = location_id # either the id of the area/station or the location of the bike
        self.bikes = []

    def get_lat(self):
        return self.lat

    def get_lon(self):
        return self.lon

    def get_location(self):
        return self.lat, self.lon

    def remove_location(self):
        self.lon = None
        self.lat = None

    def get_target_state(self, day, hour):
        return 0

    def get_arrive_intensity(self, day, hour):
        return 0

    def get_leave_intensity(self, day, hour):
        return 0

    def get_available_bikes(self):
        return []

    def get_swappable_bikes(self, battery_limit = BATTERY_LIMIT_TO_SWAP):
        return []

    def set_location(self, lat: float, lon: float, location_id):
        self.lat = lat
        self.lon = lon
        self.location_id = location_id

    def distance_to(self, lat: float, lon: float):
        """
        Returns the distance from the location to a coordinate. 
        Used in State.get_station_by_lat_lon, which returns the nearest station to the coordinates. 
        """
        return geopy.distance.distance((self.lat, self.lon), (lat, lon)).km
