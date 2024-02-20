from shapely.geometry import MultiPoint
import numpy as np
import sim
from settings import *
import copy
from sim.Location import Location

class Area(Location):
    """
    Area class representing an hexagon area on the map, with a set amount of bikes/scooters in it.
    """

    def __init__(
        self,
        area_id,
        border_vertices,
        bikes = {}, #dict, key = bike_id, value = object
        station = None,
        neighboring_areas = {},
        leave_intensities = None,
        leave_intensities_stdev = None,
        arrive_intensities = None,
        arrive_intensities_stdev = None,
        center_location = None,
        move_probabilities = None,
        target_state = None
    ):
        super().__init__(
            *(center_location if center_location else self.__compute_center(border_vertices)), area_id
        )

        self.set_bikes(bikes)

        self.station = station
        self.border_vertices = border_vertices
        self.leave_intensities = leave_intensities
        self.leave_intensities_stdev = leave_intensities_stdev
        self.arrive_intensities = arrive_intensities
        self.arrive_intensities_stdev = arrive_intensities_stdev
        self.move_probabilities = move_probabilities

        self.neighboring_areas = neighboring_areas # the area-hexes that border

        if target_state is not None:
            self.target_state = target_state
        else:
            self.target_state = [[0 for hour in range(24)] for day in range(7)]
            
        self.metrics = sim.Metric()

    def sloppycopy(self, *args):
        return Area(
            self.location_id,
            list(copy.deepcopy(self.border_vertices)),
            list(copy.deepcopy(self.bikes).values()),
            leave_intensities=self.leave_intensities,
            intensities=self.arrive_intensities,

            move_probabilities = self.move_probabilities,

            center_location = self.get_location(),
            target_state = self.target_state
        )

    def set_bikes(self, bikes):
        self.bikes = {bike.bike_id : bike for bike in bikes}

    def get_target_state(self, day, hour):
        return self.target_state[day % 7][hour % 24]

    def get_move_probabilities(self, state, day, hour):
        if self.move_probabilities is None: # if not set, it is a uniform distribution between all areas
            num_areas = len(state.areas)
            mp = [1/num_areas for _ in range(num_areas)]
            return mp
        return self.move_probabilities[day % 7][hour % 24]

    def get_arrive_intensity(self, day, hour):
        return self.arrive_intensities[day % 7][hour % 24] if self.arrive_intensities else 0

    def get_leave_intensity(self, day, hour):
        return self.leave_intensities[day % 7][hour % 24] if self.leave_intensities else 0

    def number_of_bikes(self):
        return len(self.bikes)

    def __compute_center(self, border_vertices):
        sum_lon = sum(v[0] for v in border_vertices)
        sum_lat = sum(v[1] for v in border_vertices)

        n = len(border_vertices)
        centroid_lat = sum_lat/n
        centroid_lon = sum_lon/n

        return centroid_lat, centroid_lon

    def add_bike(self, bike):
        # Adding bike to bike list
        self.bikes[bike.bike_id] = bike
        bike.set_location(self.get_lat(), self.get_lon(), self.location_id)
        return True

    def remove_bike(self, bike):
        del self.bikes[bike.bike_id]

    def get_bikes(self):
        return self.bikes.values()

    def get_available_bikes(self):
        return [
            bike for bike in self.bikes.values() if bike.usable()
        ]

    def get_swappable_bikes(self, battery_limit=70):
        """
        Filter out bikes with 100% battery and sort them by battery percentage
        """
        bikes = [
            bike for bike in self.bikes.values() if bike.hasBattery() and bike.battery < battery_limit
        ]
        return sorted(bikes, key=lambda bike: bike.battery, reverse=False)

    def get_bike_from_id(self, bike_id):
        return self.bikes[bike_id]
    
    def __repr__(self):
        return (
            f"<Station {self.location_id}: {len(self.bikes)} bikes>"
        )

    def __str__(self):
        return f"Station {self.location_id:2d}: Arrive {self.get_arrive_intensity(0, 8):4.2f} Leave {self.get_leave_intensity(0, 8):4.2f} Ideal {self.get_target_state(0, 8):3d} Bikes {len(self.bikes):3d}"
