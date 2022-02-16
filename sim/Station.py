from shapely.geometry import MultiPoint
import numpy as np
from sim.Scooter import Scooter
from sim.Location import Location
from settings import CLUSTER_CENTER_DELTA, BATTERY_LIMIT
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
    ):
        self.scooters = scooters
        self.leave_intensity_per_iteration = leave_intensity_per_iteration
        self.arrive_intensity_per_iteration = arrive_intensity_per_iteration
        self.average_number_of_scooters = average_number_of_scooters
        super().__init__(
            *(center_location if center_location else self.__compute_center()),
            cluster_id,
        )
        self.move_probabilities = move_probabilities

    # def __deepcopy__(self, *args):
    #     return Station(
    #         self.id,
    #         copy.deepcopy(self.scooters),
    #         center_location=self.get_location(),
    #         move_probabilities=self.move_probabilities,
    #         leave_intensity_per_iteration=self.leave_intensity_per_iteration,
    #         arrive_intensity_per_iteration=self.arrive_intensity_per_iteration,
    #         average_number_of_scooters=self.average_number_of_scooters,
    #     )

    class Decorators:
        @classmethod
        def check_move_probabilities(cls, func):
            def return_function(self, *args, **kwargs):
                if self.move_probabilities is not None:
                    return func(self, *args, **kwargs)
                else:
                    raise ValueError(
                        "Move probabilities matrix not initialized. Please set in the set_move_probabilities function"
                    )

            return return_function

    @Decorators.check_move_probabilities
    def get_leave_distribution(self):
        # Copy list
        distribution = self.move_probabilities.copy()
        if np.sum(distribution[np.arange(len(distribution)) != self.id]) == 0.0:
            # if all leave probabilities are zero, let them all be equally likely
            distribution = np.ones_like(distribution)
        # Set stay probability to zero
        distribution[self.id] = 0.0
        # Normalize leave distribution
        return distribution / np.sum(distribution)

    def set_move_probabilities(self, move_probabilities: np.ndarray):
        self.move_probabilities = move_probabilities

    def number_of_possible_pickups(self):
        return self.number_of_scooters()

    def number_of_scooters(self):
        return len(self.scooters)

    def __compute_center(self):
        cluster_centroid = MultiPoint(
            list(map(lambda scooter: (scooter.get_location()), self.scooters))
        ).centroid
        return cluster_centroid.x, cluster_centroid.y

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
        # Adding scooter to scooter list
        self.scooters.append(scooter)
        # Changing coordinates of scooter to this location + some delta
        delta_lat = rng.uniform(-CLUSTER_CENTER_DELTA, CLUSTER_CENTER_DELTA)
        delta_lon = rng.uniform(-CLUSTER_CENTER_DELTA, CLUSTER_CENTER_DELTA)
        scooter.set_location(self.get_lat() + delta_lat, self.get_lon() + delta_lon)

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
            raise ValueError(f"No scooters with id={scooter_id} where found")

    def __repr__(self):
        return (
            f"<Station {self.id}: {len(self.scooters)} scooters>"
        )

    def __str__(self):
        return f"Station {self.id}"
