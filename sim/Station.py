from shapely.geometry import MultiPoint
import numpy as np
from sim.Scooter import Scooter
from sim.Location import Location
from settings import STATION_CENTER_DELTA, BATTERY_LIMIT, DEFAULT_STATION_CAPACITY
import copy

class Station(Location):
    """
    Station class representing a collection of e-scooters. Contains all customer behaviour data.
    """

    def __init__(
        self,
        station_id,
        scooters = [],
        leave_intensity_per_iteration=None,
        arrive_intensity_per_iteration=None,
        center_location=None,
        move_probabilities=None,
        average_number_of_scooters=None,
        target_state=None,
        capacity=DEFAULT_STATION_CAPACITY,
        original_id = None,
        charging_station = None,
    ):
        super().__init__(
            *(center_location if center_location else self.__compute_center(scooters)),
            station_id, target_state=target_state
        )
        self.set_scooters(scooters)
        self.leave_intensity_per_iteration = leave_intensity_per_iteration
        self.arrive_intensity_per_iteration = arrive_intensity_per_iteration
        self.average_number_of_scooters = average_number_of_scooters
        self.move_probabilities = move_probabilities
        self.capacity = int(capacity)
        self.original_id = original_id
        self.charging_station = charging_station
        
        self.net_demand = 0 #needed for gleditsch_hagen_policy
        self.pickup_station = 0 #needed for gleditsch_hagen_policy
        self.deviation_not_visited = 0 #needed for gleditsch_hagen_policy
        self.time_to_violation = 0  #needed for gleditsch_hagen_policy
        self.base_violations = 0  #needed for gleditsch_hagen_policy
        self.target state = 0  #needed for gleditsch_hagen_policy
        
        if len(self.scooters) > self.capacity:
            self.capacity = len(self.scooters)

    def sloppycopy(self, *args):
        return Station(
            self.id,
            list(copy.deepcopy(self.scooters).values()),
            leave_intensity_per_iteration=self.leave_intensity_per_iteration,
            arrive_intensity_per_iteration=self.arrive_intensity_per_iteration,
            center_location=self.get_location(),
            move_probabilities=self.move_probabilities,
            average_number_of_scooters=self.average_number_of_scooters,
            target_state=self.target_state,
            capacity=self.capacity,
            original_id=self.original_id,
            charging_station=self.charging_station,
        )

    def set_scooters(self, scooters):
        self.scooters = {scooter.id : scooter for scooter in scooters}

    def spare_capacity(self):
        return self.capacity - len(self.scooters)

    def get_leave_distribution(self, state, day, hour):
        if self.move_probabilities is None:
            mp = []
            for station in range(len(state.locations)):
                mp.append(1 / len(state.locations))
            return mp
        return self.move_probabilities[day % 7][hour % 24]

    def get_arrive_intensity(self, day, hour):
        return self.arrive_intensity_per_iteration[day % 7][hour % 24]

    def get_leave_intensity(self, day, hour):
        return self.leave_intensity_per_iteration[day % 7][hour % 24]

    def get_target_state(self, day, hour):
        if self.target_state:
            return self.target_state[day % 7][hour % 24]
        else:
            return 0

    def number_of_scooters(self):
        return len(self.scooters)

    def __compute_center(self, scooters):
        if len(scooters) > 0:
            station_centroid = MultiPoint(
                list(map(lambda scooter: (scooter.get_location()), scooters))
            ).centroid
            return station_centroid.x, station_centroid.y
        else:
            return 0, 0

    def add_scooter(self, rng, scooter: Scooter):
        if len(self.scooters) >= self.capacity:
            return False
        # Adding scooter to scooter list
        self.scooters[scooter.id] = scooter
        # Changing coordinates of scooter to this location + some delta
        delta_lat = rng.uniform(-STATION_CENTER_DELTA, STATION_CENTER_DELTA)
        delta_lon = rng.uniform(-STATION_CENTER_DELTA, STATION_CENTER_DELTA)
        scooter.set_location(self.get_lat() + delta_lat, self.get_lon() + delta_lon)
        return True

    def remove_scooter(self, scooter: Scooter):
        del self.scooters[scooter.id]

    def get_scooters(self):
        return self.scooters.values()

    def get_available_scooters(self):
        return [
            scooter for scooter in self.scooters.values() if scooter.usable()
        ]

    def get_swappable_scooters(self, battery_limit=70):
        """
        Filter out scooters with 100% battery and sort them by battery percentage
        """
        scooters = [
            scooter for scooter in self.scooters.values() if scooter.hasBattery() and scooter.battery < battery_limit
        ]
        return sorted(scooters, key=lambda scooter: scooter.battery, reverse=False)

    def get_scooter_from_id(self, scooter_id):
        return self.scooters[scooter_id]

    def __repr__(self):
        return (
            f"<Station {self.id}: {len(self.scooters)} scooters>"
        )

    def __str__(self):
        return f"Station {self.id:2d}: Arrive {self.get_arrive_intensity(0, 8):4.2f} Leave {self.get_leave_intensity(0, 8):4.2f} Ideal {self.get_target_state(0, 8):3d} Scooters {len(self.scooters):3d}"
