from shapely.geometry import MultiPoint
import numpy as np
from sim.Location import Location
import sim
from settings import STATION_CENTER_DELTA, BATTERY_LIMIT, DEFAULT_STATION_CAPACITY
import copy

class Station(Location):
    """
    Station class representing a collection of bikes. Contains all customer behaviour data.
    """

    def __init__(
        self,
        station_id,
        bikes = [],
        leave_intensities=None,
        leave_intensities_stdev=None,
        arrive_intensities=None,
        arrive_intensities_stdev=None,
        center_location=None,
        move_probabilities=None,
        average_number_of_bikes=None,
        target_state=None,
        capacity=DEFAULT_STATION_CAPACITY,
        original_id = None,
        charging_station = None,
    ):
        super().__init__(
            *(center_location if center_location else self.__compute_center(bikes)), station_id
        )

        self.set_bikes(bikes)

        self.historical_leave_intensities = leave_intensities
        self.historical_leave_intensities_stdev = leave_intensities_stdev
        self.historical_arrive_intensities = arrive_intensities
        self.historical_arrive_intensities_stdev = arrive_intensities_stdev

        self.move_probabilities = move_probabilities

        self.average_number_of_bikes = average_number_of_bikes
        self.capacity = int(capacity)
        self.original_id = original_id
        self.charging_station = charging_station

        if target_state is not None:
            self.target_state = target_state
        else:
            self.target_state = []
            for day in range(7):
                self.target_state.append([])
                for hour in range(24):
                    self.target_state[day].append(0)
            
        self.metrics = sim.Metric()

        if len(self.bikes) > self.capacity:
            self.capacity = len(self.bikes)

    def sloppycopy(self, *args):
        return Station(
            self.id,
            list(copy.deepcopy(self.bikes).values()),

            historical_leave_intensities=self.historical_leave_intensities,
            historical_leave_intensities_stdev=self.historical_leave_intensities_stdev,
            historical_arrive_intensities=self.historical_arrive_intensities,
            historical_arrive_intensities_stdev=self.historical_arrive_intensities_stdev,

            leave_intensities=self.leave_intensities,
            arrive_intensities=self.arrive_intensities,

            move_probabilities=self.move_probabilities,

            center_location=self.get_location(),
            average_number_of_bikes=self.average_number_of_bikes,
            target_state=self.target_state,
            capacity=self.capacity,
            original_id=self.original_id,
            charging_station=self.charging_station,
        )

    def set_bikes(self, bikes):
        self.bikes = {bike.id : bike for bike in bikes}

    def spare_capacity(self):
        return self.capacity - len(self.bikes)

    def get_target_state(self, day, hour):
        return self.target_state[day % 7][hour % 24]

    def get_leave_distribution(self, state, day, hour):
        if self.move_probabilities is None:
            mp = []
            for station in range(len(state.locations)):
                mp.append(1 / len(state.locations))
            return mp
        return self.move_probabilities[day % 7][hour % 24]

    def get_arrive_intensity(self, day, hour):
        return self.arrive_intensities[day % 7][hour % 24]

    def get_leave_intensity(self, day, hour):
        return self.leave_intensities[day % 7][hour % 24]

    def number_of_bikes(self):
        return len(self.bikes)

    def __compute_center(self, bikes):
        if len(bikes) > 0:
            station_centroid = MultiPoint(
                list(map(lambda bike: (bike.get_location()), bikes))
            ).centroid
            return station_centroid.x, station_centroid.y
        else:
            return 0, 0

    def add_bike(self, bike):
        if len(self.bikes) >= self.capacity:
            return False
        # Adding bike to bike list
        self.bikes[bike.id] = bike
        bike.set_location(self.get_lat(), self.get_lon())
        return True

    def remove_bike(self, bike):
        del self.bikes[bike.id]

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
            f"<Station {self.id}: {len(self.bikes)} bikes>"
        )

    def __str__(self):
        return f"Station {self.id:2d}: Arrive {self.get_arrive_intensity(0, 8):4.2f} Leave {self.get_leave_intensity(0, 8):4.2f} Ideal {self.get_target_state(0, 8):3d} Bikes {len(self.bikes):3d}"
