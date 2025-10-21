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
        neighbours = [],
        leave_intensities = None,
        leave_intensities_stdev = None,
        arrive_intensities = None,
        arrive_intensities_stdev = None,
        center_location = None,
        move_probabilities = None,
        target_state = None,
        is_station_based = False
    ):
        super().__init__(
            *(center_location if center_location else self.__compute_center(border_vertices)), area_id
        )

        self.set_bikes(bikes)

        self.station = station
        self.border_vertices = border_vertices
        self.leave_intensities = leave_intensities if leave_intensities else [[0 for _ in range(24)] for _ in range(7)] 
        self.leave_intensities_stdev = leave_intensities_stdev if leave_intensities_stdev else [[0 for _ in range(24)] for _ in range(7)] 
        self.arrive_intensities = arrive_intensities if arrive_intensities else [[0 for _ in range(24)] for _ in range(7)] 
        self.arrive_intensities_stdev = arrive_intensities_stdev if arrive_intensities_stdev else [[0 for _ in range(24)] for _ in range(7)] 

        self.move_probabilities = move_probabilities
        self.is_station_based = is_station_based

        self.neighbours = neighbours

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

    def get_difference_from_target(self, day, hour, with_neighbours):
        if with_neighbours:
            num_bikes = sum(neighbor.number_of_bikes() for neighbor in self.neighbours) + self.number_of_bikes()
            sum_target = sum(neighbor.get_target_state(day, hour) for neighbor in self.neighbours) + self.get_target_state(day, hour)
            return num_bikes - sum_target
        return self.number_of_bikes() - self.get_target_state(day, hour)
    
    def get_difference_from_target_discounted_drive_time(self, day, hour, depature_area, simul):
        travel_time = simul.state.get_vehicle_travel_time(depature_area.location_id, self.location_id)
        diff = self.get_difference_from_target(day, hour, True)
        return diff * (1 - (travel_time/30))

    def set_bikes(self, bikes):
        self.bikes = {bike.bike_id : bike for bike in bikes}
        for bike in bikes:
            bike.set_location(self.lat, self.lon, self.location_id)

    def get_neighbours(self):
        return self.neighbours

    def get_target_state(self, day, hour):
        return self.target_state[day % 7][hour % 24]

    def get_move_probabilities(self, state, day, hour):
        if self.move_probabilities is None: # if not set, it is a uniform distribution between all areas
            num_areas = len(state.areas)
            mp = {area.location_id: 1/num_areas for area in state.get_areas()}
            return mp
        mp = self.move_probabilities[day % 7][hour % 24]
        mp2 = {area.location_id: 0.0 for area in state.get_areas()}
        for key, value in mp.items():
            if key in mp2.keys():
                mp2[key] = value
        return mp2

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
        # bike.set_location(None, None, None)

    def get_bikes(self):
        return list(self.bikes.values())

    def get_available_bikes(self):
        return [
            bike for bike in self.bikes.values() if bike.usable()
        ]

    def get_unusable_bikes(self):
        return [
            bike for bike in self.bikes.values() if not bike.usable()
        ]

    def get_swappable_bikes(self, battery_limit=BATTERY_LIMIT_TO_SWAP):
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
            f"<Area {self.location_id}: {len(self.bikes)} bikes>"
        )

    def __str__(self):
        return f"Area {self.location_id}: Arrive {self.get_arrive_intensity(0, 8):4.2f} Leave {self.get_leave_intensity(0, 8):4.2f} Ideal {self.get_target_state(0, 8):4.2f} Bikes {len(self.bikes):3d}"
