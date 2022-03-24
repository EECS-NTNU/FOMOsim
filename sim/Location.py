from math import sqrt, pi, sin, cos, atan2
from settings import GEOSPATIAL_BOUND_NEW


class Location:
    """
    Base location class. All classes representing a geographic position inherit from the Location class
    """

    def __init__(self, lat: float, lon: float, location_id: int, ideal_state = 0):
        self.lat = lat
        self.lon = lon
        self.id = location_id
        self.ideal_state = ideal_state

    def get_lat(self):
        return self.lat

    def get_lon(self):
        return self.lon

    def get_location(self):
        return self.lat, self.lon

    def remove_location(self):
        self.lon = None
        self.lat = None

    def set_location(self, lat: float, lon: float):
        lat_min, lat_max, lon_min, lon_max = GEOSPATIAL_BOUND_NEW
        if lat is None:
            self.lat = lat
        elif lat < lat_min:
            self.lat = lat_min
        elif lat > lat_max:
            self.lat = lat_max
        else:
            self.lat = lat

        if lon is None:
            self.lon = lon
        elif lon < lon_min:
            self.lon = lon_min
        elif lon > lon_max:
            self.lon = lon_max
        else:
            self.lon = lon

    def distance_to(self, lat: float, lon: float):
        return Location.haversine(self.lat, self.lon, lat, lon)

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """
        Compute the distance between two points in meters
        :param lat1: Coordinate 1 lat
        :param lon1: Coordinate 1 lon
        :param lat2: Coordinate 2 lat
        :param lon2: Coordinate 2 lon
        :return: Kilometers between coordinates
        """
        radius = 6378.137
        d_lat = lat2 * pi / 180 - lat1 * pi / 180
        d_lon = lon2 * pi / 180 - lon1 * pi / 180
        a = sin(d_lat / 2) * sin(d_lat / 2) + cos(lat1 * pi / 180) * cos(
            lat2 * pi / 180
        ) * sin(d_lon / 2) * sin(d_lon / 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = radius * c
        return distance
