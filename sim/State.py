from sim.Location import Location
from sim.Station import Station
from sim.Depot import Depot
import clustering.methods
from sim.SaveMixin import SaveMixin
from visualization.visualizer import *
import numpy as np
import math
from settings import STATE_CACHE_DIR
import copy


class State(SaveMixin):
    """
    Container class for the whole state of all clusters. Data concerning the interplay between clusters are stored here
    """

    def __init__(
        self,
        stations: [Station] = [],
        depots: [Depot] = [],
        vehicles: [Vehicle] = [],
        scooters_in_use: [Location] = [], # scooters in use (not parked at any station)
        distance_matrix=None, # will calculate based on coordinates if not given
        rng = None,
    ):
        if rng is None:
            self.rng = np.random.default_rng(None)
        else:
            self.rng = rng

        self.stations = stations
        self.vehicles = vehicles
        self.depots = depots
        self.scooters_in_use = scooters_in_use

        self.locations = self.stations + self.depots
        if distance_matrix:
            self.distance_matrix = distance_matrix
        else:
            self.distance_matrix = self.calculate_distance_matrix()
        self.TRIP_INTENSITY_RATE = 0.1

    # def __deepcopy__(self, *args):
    #     new_state = State(
    #         copy.deepcopy(self.stations),
    #         copy.deepcopy(self.depots),
    #         copy.deepcopy(self.vehicles),
    #         distance_matrix=self.distance_matrix,
    #     )
    #     for vehicle in new_state.vehicles:
    #         vehicle.current_location = new_state.get_location_by_id(
    #             vehicle.current_location.id
    #         )
    #     return new_state

    def scooter_in_use(self, scooter):
        self.scooters_in_use.append(scooter)

    def remove_used_scooter(self, scooter):
        self.scooters_in_use.remove(scooter)

    def get_used_scooter(self):
        if len(self.scooters_in_use) > 0:
            return self.scooters_in_use.pop()

    def get_all_locations(self):
        return self.locations

    def get_cluster_by_lat_lon(self, lat: float, lon: float):
        """
        :param lat: lat location of scooter
        :param lon:
        :return:
        """
        return min(self.stations, key=lambda cluster: cluster.distance_to(lat, lon))

    # parked scooters
    def get_scooters(self):
        all_scooters = []
        for cluster in self.stations:
            for scooter in cluster.scooters:
                all_scooters.append(scooter)
        return all_scooters

    # parked and in-use scooters
    def get_all_scooters(self):
        all_scooters = []
        for cluster in self.stations:
            for scooter in cluster.scooters:
                all_scooters.append(scooter)
        all_scooters.extend(self.scooters_in_use)
        return all_scooters

    def sample_size(self):
        return len(self.scooters_in_use) + len(self.get_scooters())

    def get_distance(self, start_location_id: int, end_location_id: int):
        """
        Calculate distance between two stations
        :param start_location_id: Location id
        :param end_location_id: Location id
        :return: float - distance in kilometers
        """
        return self.distance_matrix[start_location_id][end_location_id]

    def get_distance_to_all_clusters(self, location_id):
        return self.distance_matrix[location_id][: len(self.stations)]

    def calculate_distance_matrix(self):
        """
        Computes distance matrix for all stations
        :return: Distance matrix
        """
        distance_matrix = []
        for location in self.locations:
            neighbour_distance = []
            for neighbour in self.locations:
                if location == neighbour:
                    neighbour_distance.append(0.0)
                else:
                    neighbour_distance.append(
                        location.distance_to(*neighbour.get_location())
                    )
            distance_matrix.append(neighbour_distance)
        return distance_matrix

    def do_action(self, action: Action, vehicle: Vehicle, time: int):
        """
        Performs an action on the state -> changing the state
        :param time: at what time the action is performed
        :param vehicle: Vehicle to perform this action
        :param action: Action - action to be performed on the state
        """
        refill_time = 0
        if vehicle.is_at_depot():
            batteries_to_swap = min(
                vehicle.flat_batteries(),
                vehicle.current_location.get_available_battery_swaps(time),
            )

            refill_time += vehicle.current_location.swap_battery_inventory(
                time, batteries_to_swap
            )
            vehicle.add_battery_inventory(batteries_to_swap)

        else:
            # Perform all pickups
            for pick_up_scooter_id in action.pick_ups:
                pick_up_scooter = vehicle.current_location.get_scooter_from_id(
                    pick_up_scooter_id
                )
                # Picking up scooter and adding to vehicle inventory and swapping battery
                vehicle.pick_up(pick_up_scooter)

                # Remove scooter from current cluster
                vehicle.current_location.remove_scooter(pick_up_scooter)
            # Perform all battery swaps
            for battery_swap_scooter_id in action.battery_swaps:
                battery_swap_scooter = vehicle.current_location.get_scooter_from_id(
                    battery_swap_scooter_id
                )
                # Decreasing vehicle battery inventory
                vehicle.change_battery(battery_swap_scooter)

            # Dropping of scooters
            for delivery_scooter_id in action.delivery_scooters:
                # Removing scooter from vehicle inventory
                delivery_scooter = vehicle.drop_off(delivery_scooter_id)

                # Adding scooter to current cluster and changing coordinates of scooter
                vehicle.current_location.add_scooter(self.rng, delivery_scooter)

        # Moving the state/vehicle from this to next cluster
        vehicle.set_current_location(
            self.get_location_by_id(action.next_location), action
        )

        return refill_time

    def __repr__(self):
        string = f"<State: {len(self.get_scooters())} scooters in {len(self.stations)} stations with {len(self.vehicles)} vehicles>\n"
        for station in self.stations:
            string += f"{repr(station)}\n"
        string += f"In use: {self.scooters_in_use}"
        return string

    def get_neighbours(
        self,
        location: Location,
        number_of_neighbours=None,
        is_sorted=True,
        exclude=None,
    ):
        """
        Get sorted list of stations closest to input cluster
        :param is_sorted: Boolean if the neighbours list should be sorted in a ascending order based on distance
        :param location: location to find neighbours for
        :param number_of_neighbours: number of neighbours to return
        :param exclude: neighbor ids to exclude
        :return:
        """
        neighbours = [
            state_location
            for state_location in self.locations
            if state_location.id != location.id
            and state_location.id not in (exclude if exclude else [])
        ]
        if is_sorted:
            neighbours = sorted(
                [
                    state_location
                    for state_location in self.locations
                    if state_location.id != location.id
                ],
                key=lambda state_location: self.distance_matrix[location.id][
                    state_location.id
                ],
            )
        return neighbours[:number_of_neighbours] if number_of_neighbours else neighbours

    def get_location_by_id(self, location_id: int):
        matches = [
            location for location in self.locations if location_id == location.id
        ]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"There are more than one location ({len(matches)} locations) matching on id {location_id} in this state"
            )
        else:
            raise ValueError(f"No locations with id={location_id} where found")

    def visualize(self):
        visualize_state(self)

    def visualize_clustering(self):
        visualize_clustering(self.stations)

    def visualize_flow(
        self,
        flows: [(int, int, int)],
    ):
        visualize_cluster_flow(self, flows)

    def visualize_action(
        self,
        vehicle_before_action: Vehicle,
        current_state: State,
        vehicle: Vehicle,
        action: Action,
        world_time,
        action_time,
        scooter_battery: bool,
        policy: str,
    ):
        visualize_action(
            self,
            vehicle_before_action,
            current_state,
            vehicle,
            action,
            world_time,
            action_time,
            scooter_battery,
            policy,
        )

    def visualize_vehicle_routes(
        self,
        current_vehicle_id: int,
        current_location_id: int,
        next_location_id: int,
        tabu_list: [int],
        policy: str,
    ):
        visualize_vehicle_routes(
            self,
            current_vehicle_id,
            current_location_id,
            next_location_id,
            tabu_list,
            policy,
        )

    def visualize_current_trips(self, trips: [(int, int, int)]):
        visualize_scooters_on_trip(self, trips)

    def visualize_system_simulation(self, trips):
        visualize_scooter_simulation(self, trips)

    def set_probability_matrix(self, probability_matrix: np.ndarray):
        if probability_matrix.shape != (len(self.stations), len(self.stations)):
            ValueError(
                f"The shape of the probability matrix does not match the number of stations in the class:"
                f" {probability_matrix.shape} != {(len(self.stations), len(self.stations))}"
            )
        for cluster in self.stations:
            cluster.set_move_probabilities(probability_matrix[cluster.id])

    def save_state(self):
        super().save(STATE_CACHE_DIR)

    @staticmethod
    def save_path(
        number_of_stations,
        sample_size,
    ):
        def convert_binary(binary):
            return 1 if binary else 0

        return (
            f"c{number_of_stations}s{sample_size}_"
        )

    def get_filename(self):
        return State.save_path(
            len(self.stations),
            len(self.get_scooters())
        )

    def sample(self, sample_size: int):
        # Filter out scooters not in sample
        sampled_scooter_ids = self.rng.choice(
            [scooter.id for scooter in self.get_scooters()], sample_size, replace=False,
        )
        for cluster in self.stations:
            cluster.scooters = [
                scooter
                for scooter in cluster.scooters
                if scooter.id in sampled_scooter_ids
            ]

    def get_random_cluster(self, exclude=None):
        return rng.choice(
            [cluster for cluster in self.stations if cluster.id != exclude.id]
            if exclude
            else self.stations
        )

    def get_vehicle_by_id(self, vehicle_id: int) -> Vehicle:
        """
        Returns the vehicle object in the state corresponding to the vehicle id input
        :param vehicle_id: the id of the vehicle to fetch
        :return: vehicle object
        """
        try:
            return [vehicle for vehicle in self.vehicles if vehicle_id == vehicle.id][0]
        except IndexError:
            raise ValueError(
                f"There are no vehicle in the state with an id of {vehicle_id}"
            )
