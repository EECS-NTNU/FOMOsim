from shapely.geometry import MultiPoint
import numpy as np
from sim.Scooter import Scooter
from sim.Location import Location
from settings import CLUSTER_CENTER_DELTA, BATTERY_LIMIT, DEFAULT_STATION_CAPACITY
import copy

class Station(Location):
    """
    Station class representing a collection of e-scooters. Contains all customer behaviour data.
    """

    def __init__(
        self,
        cluster_id: int,
        scooters: [Scooter],
        leave_intensity_per_iteration=None,
        arrive_intensity_per_iteration=None,
        center_location=None,
        move_probabilities=None,
        average_number_of_scooters=None,
        target_state=None,
        capacity=DEFAULT_STATION_CAPACITY,
        original_id = None,
    ):
        super().__init__(
            *(center_location if center_location else self.__compute_center(scooters)),
            cluster_id, target_state=target_state
        )
        self.scooters = scooters
        self.leave_intensity_per_iteration = leave_intensity_per_iteration
        self.arrive_intensity_per_iteration = arrive_intensity_per_iteration
        self.average_number_of_scooters = average_number_of_scooters
        self.move_probabilities = move_probabilities
        self.capacity = capacity
        self.original_id = original_id

    def sloppycopy(self, *args):
        return Station(
            self.id,
            copy.deepcopy(self.scooters),
            leave_intensity_per_iteration=self.leave_intensity_per_iteration,
            arrive_intensity_per_iteration=self.arrive_intensity_per_iteration,
            center_location=self.get_location(),
            move_probabilities=self.move_probabilities,
            average_number_of_scooters=self.average_number_of_scooters,
            target_state=self.target_state,
            capacity=self.capacity,
        )

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

    def number_of_possible_pickups(self):
        return self.number_of_scooters()

    def number_of_scooters(self):
        return len(self.scooters)

    def __compute_center(self, scooters):
        if len(scooters) > 0:
            cluster_centroid = MultiPoint(
                list(map(lambda scooter: (scooter.get_location()), scooters))
            ).centroid
            return cluster_centroid.x, cluster_centroid.y
        else:
            return 0, 0

    def add_scooter(self, rng, scooter: Scooter):
        matches = [
            cluster_scooter
            for cluster_scooter in self.scooters
            if scooter.id == cluster_scooter.id
        ]
        if len(matches) > 0:
            raise ValueError(
                f"The scooter you are trying to add is already in the cluster: {matches}"
            )
        if len(self.scooters) >= self.capacity:
            return False
        # Adding scooter to scooter list
        self.scooters.append(scooter)
        # Changing coordinates of scooter to this location + some delta
        delta_lat = rng.uniform(-CLUSTER_CENTER_DELTA, CLUSTER_CENTER_DELTA)
        delta_lon = rng.uniform(-CLUSTER_CENTER_DELTA, CLUSTER_CENTER_DELTA)
        scooter.set_location(self.get_lat() + delta_lat, self.get_lon() + delta_lon)
        return True

    def remove_scooter(self, scooter: Scooter):
        self.scooters.remove(scooter)

    def get_available_scooters(self):
        return [
            scooter for scooter in self.scooters if scooter.usable()
        ]

    def print_all_scooters(self, with_coordinates=False):
        string = ""
        for scooter in self.scooters:
            string += f"ID: {scooter.id}  Battery {round(scooter.battery, 1)}"
            string += (
                f"Coord: {scooter.get_location()} | " if with_coordinates else " | "
            )
        return string if string != "" else "Empty cluster"

    def get_swappable_scooters(self, battery_limit=70):
        """
        Filter out scooters with 100% battery and sort them by battery percentage
        """
        scooters = [
            scooter for scooter in self.scooters if scooter.hasBattery() and scooter.battery < battery_limit
        ]
        return sorted(scooters, key=lambda scooter: scooter.battery, reverse=False)

    def get_scooter_from_id(self, scooter_id):
        matches = [
            cluster_scooter
            for cluster_scooter in self.scooters
            if scooter_id == cluster_scooter.id
        ]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"There are more than one scooter ({len(matches)} scooters) "
                f"matching on id {scooter_id} in Station {self.id}"
            )
        else:
            raise ValueError(f"No scooters with id={scooter_id} were found")

    def __repr__(self):
        return (
            f"<Station {self.id}: {len(self.scooters)} scooters>"
        )

    def __str__(self):
        return f"Station {self.id:2d}: Arrive {self.get_arrive_intensity(0, 8):4.2f} Leave {self.get_leave_intensity(0, 8):4.2f} Ideal {self.get_target_state(0, 8):3d} Scooters {len(self.scooters):3d}"
