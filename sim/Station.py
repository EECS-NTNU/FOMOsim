from shapely.geometry import MultiPoint
import numpy as np
from sim.Location import Location
import sim
from settings import *
import copy

class Station(Location):
    """
    Station class representing a collection of bikes. Contains all customer behaviour data.
    """

    def __init__(
        self,
        station_id,
        bikes = {},
        leave_intensities=None,
        leave_intensities_stdev=None,
        arrive_intensities=None,
        arrive_intensities_stdev=None,
        center_location=None,
        move_probabilities=None,
        average_number_of_bikes=None,
        target_state=None,
        capacity = DEFAULT_STATION_CAPACITY,
        original_id = None,
        charging_station = None, 
    ):
        super().__init__(
            *(center_location if center_location else self.__compute_center(bikes)), station_id
        )

        self.set_bikes(bikes)

        self.leave_intensities = leave_intensities if leave_intensities else [[0 for _ in range(24)] for _ in range(7)] 
        self.leave_intensities_stdev = leave_intensities_stdev if leave_intensities_stdev else [[0 for _ in range(24)] for _ in range(7)] 
        self.arrive_intensities = arrive_intensities if arrive_intensities else [[0 for _ in range(24)] for _ in range(7)] 
        self.arrive_intensities_stdev = arrive_intensities_stdev if arrive_intensities_stdev else [[0 for _ in range(24)] for _ in range(7)] 

        self.move_probabilities = move_probabilities

        self.average_number_of_bikes = average_number_of_bikes
        self.capacity = int(capacity) if capacity != 'inf' else float(capacity) # handles if capacity isn't infinite
        self.original_id = original_id
        self.charging_station = charging_station
        self.neighboring_stations = []

        if target_state is not None:
            self.target_state = target_state
        else:
            self.target_state = [[0 for hour in range(24)] for day in range(7)]
            
        self.metrics = sim.Metric()

        if len(self.bikes) > self.capacity:
            self.capacity = len(self.bikes)

    def sloppycopy(self, *args):
        return Station(
            self.location_id,
            list(copy.deepcopy(self.bikes).values()),

            leave_intensities=self.leave_intensities,
            leave_intensities_stdev=self.leave_intensities_stdev,
            arrive_intensities=self.arrive_intensities,
            arrive_intensities_stdev=self.arrive_intensities_stdev,

            move_probabilities=self.move_probabilities,

            center_location=self.get_location(),
            average_number_of_bikes=self.average_number_of_bikes,
            target_state=self.target_state,
            capacity=self.capacity,
            original_id=self.original_id,
            charging_station=self.charging_station,
        )

    def set_bikes(self, bikes):
        self.bikes = {bike.bike_id : bike for bike in bikes}

    def spare_capacity(self):
        return self.capacity - len(self.bikes)
    
    def get_target_state(self, day, hour):
        return self.target_state[day % 7][hour % 24]

    def get_move_probabilities(self, state, day, hour):
        """
        Returns a dictionary. Key = location_id, Value = probability to go there
        """
        if self.move_probabilities is None:
            num_stations = len(state.stations)
            mp = {station_id: 1/num_stations for station_id in state.get_station_ids()}
            return mp
        return self.move_probabilities[day % 7][hour % 24]

    def get_arrive_intensity(self, day, hour):
        return self.arrive_intensities[day % 7][hour % 24]

    def get_leave_intensity(self, day, hour):
        return self.leave_intensities[day % 7][hour % 24]

    def get_arrive_intensity_stdev(self, day, hour):
        return self.arrive_intensities_stdev[day % 7][hour % 24]
    
    def get_leave_intensity_stdev(self, day, hour):
        return self.leave_intensities_stdev[day % 7][hour % 24]

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
        self.bikes[bike.bike_id] = bike
        bike.set_location(self.get_lat(), self.get_lon(), self.location_id)
        return True

    def remove_bike(self, bike):
        del self.bikes[bike.bike_id]

    def get_bikes(self):
        return list(self.bikes.values())

    def get_available_bikes(self):
        return [
            bike for bike in self.bikes.values() if bike.usable()
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
    
    def set_neighboring_stations(self, neighboring_stations_dict, location_list):
        """
        Defines the list neighboring_stations consisting of Station-objects
        """
        self_index = location_list.index(self)
        neighboring_stations_list = neighboring_stations_dict[self_index]
        for index in neighboring_stations_list:
            self.neighboring_stations.append(location_list[index])
        return None
    
    def set_move_probabilities(self, station_list):
        move_probabilities = [[{} for _ in range(24)] for _ in range(7)]
        for day in range(7):
            for hour in range(24):
                for ind in range(len(self.move_probabilities[day][hour])):
                    station_id = station_list[ind].location_id
                    move_probabilities[day][hour][station_id] = self.move_probabilities[day][hour][ind]
        
        self.move_probabilities = move_probabilities

    def __repr__(self):
        return (
            f"<Station {self.location_id}: {len(self.bikes)} bikes>"
        )

    def __str__(self):
        return f"Station {self.location_id:2d}: Arrive {self.get_arrive_intensity(0, 8):4.2f} Leave {self.get_leave_intensity(0, 8):4.2f} Ideal {self.get_target_state(0, 8):3d} Bikes {len(self.bikes):3d}"
