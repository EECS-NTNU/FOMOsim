import random
from sim.Location import Location
from sim.Station import Station
from sim.Depot import Depot
import clustering.methods
from sim.SaveMixin import SaveMixin
from visualization.visualizer import *
import policies.neighbour_filtering
import numpy as np
import math
from globals import STATE_CACHE_DIR
import copy


class State(SaveMixin):
    """
    Container class for the whole state of all clusters. Data concerning the interplay between clusters are stored here
    """

    def __init__(
        self,
        stations: [Station],
        depots: [Depot],
        vehicles: [Vehicle],
        distance_matrix=None, # will calculate based on coordinates if not given
    ):
        self.stations = stations
        self.vehicles = vehicles
        self.depots = depots
        self.locations = self.stations + self.depots
        if distance_matrix:
            self.distance_matrix = distance_matrix
        else:
            self.distance_matrix = self.calculate_distance_matrix()
        self.simulation_scenarios = None
        self.TRIP_INTENSITY_RATE = 0.1

    def __deepcopy__(self, *args):
        new_state = State(
            copy.deepcopy(self.stations),
            copy.deepcopy(self.depots),
            copy.deepcopy(self.vehicles),
            distance_matrix=self.distance_matrix,
        )
        new_state.simulation_scenarios = self.simulation_scenarios
        for vehicle in new_state.vehicles:
            vehicle.current_location = new_state.get_location_by_id(
                vehicle.current_location.id
            )
        return new_state

    def get_all_locations(self):
        return self.locations

    def get_cluster_by_lat_lon(self, lat: float, lon: float):
        """
        :param lat: lat location of scooter
        :param lon:
        :return:
        """
        return min(self.stations, key=lambda cluster: cluster.distance_to(lat, lon))

    def get_scooters(self):
        all_scooters = []
        for cluster in self.stations:
            for scooter in cluster.scooters:
                all_scooters.append(scooter)
        return all_scooters

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

    def get_possible_actions(
        self,
        vehicle: Vehicle,
        number_of_neighbours,
        divide=None,
        exclude=None,
        time=None,
    ):
        """
        Enumerate all possible actions from the current state
        :param time: time of the world when the actions is to be performed
        :param exclude: stations to exclude from next cluster
        :param vehicle: vehicle to perform this action
        :param number_of_neighbours: number of neighbours to evaluate, if None: all neighbors are returned
        :param divide: number to divide by to create range increment
        :return: List of Action objects
        """
        actions = []
        neighbours = policies.neighbour_filtering.filtering_neighbours(
            self,
            vehicle,
            0,
            0,
            number_of_neighbours,
            exclude=exclude,
        )
        # Return empty action if
        if not vehicle.is_at_depot():

            def get_range(max_int):
                if divide and divide > 0 and max_int > 0:
                    return list(
                        {
                            *(
                                [
                                    i
                                    for i in range(
                                        0, int(max_int + 1), int(math.ceil(max_int / divide))
                                    )
                                ]
                                + [int(max_int)]
                            )
                        }
                    )
                else:
                    return [i for i in range(int(max_int + 1))]

            # Initiate constraints for battery swap, pick-up and drop-off
            pick_ups = min(
                len(vehicle.current_location.scooters),
                vehicle.scooter_inventory_capacity - len(vehicle.scooter_inventory),
                vehicle.battery_inventory,
            )
            swaps = vehicle.get_max_number_of_swaps()
            drop_offs = max(
                min(
                    len(vehicle.current_location.scooters),
                    len(vehicle.scooter_inventory),
                ),
                0,
            )
            combinations = []
            # Different combinations of battery swaps, pick-ups, drop-offs and stations
            for pick_up in get_range(pick_ups):
                for swap in get_range(swaps):
                    for drop_off in get_range(drop_offs):
                        if (
                            (pick_up + swap) <= len(vehicle.current_location.scooters)
                            and (pick_up + swap) <= vehicle.battery_inventory
                            and (pick_up + swap + drop_off > 0)
                        ):
                            for (
                                location
                            ) in policies.neighbour_filtering.filtering_neighbours(
                                self,
                                vehicle,
                                pick_up,
                                drop_off,
                                number_of_neighbours,
                                exclude=exclude,
                            ):
                                combinations.append(
                                    [
                                        max(
                                            min(
                                                vehicle.battery_inventory - pick_up,
                                                swap,
                                            ),
                                            0,
                                        ),
                                        pick_up,
                                        drop_off,
                                        location.id,
                                    ]
                                )

            # Assume that no battery swap or pick-up of scooters with 100% battery and
            # that the scooters with the lowest battery are prioritized
            swappable_scooters_id = [
                scooter.id
                for scooter in vehicle.current_location.get_swappable_scooters()
            ]

            none_swappable_scooters_id = [
                scooter.id
                for scooter in vehicle.current_location.scooters
                if scooter.battery >= 70
            ]

            def choose_pick_up(swaps, pickups):
                number_of_none_swappable_scooters = max(
                    swaps + pickups - len(swappable_scooters_id), 0
                )

                return (
                    swappable_scooters_id[swaps : swaps + pick_up]
                    + none_swappable_scooters_id[:number_of_none_swappable_scooters]
                )

            # Adding every action. Actions are the IDs of the scooters to be handled.
            for battery_swap, pick_up, drop_off, cluster_id in combinations:
                if not vehicle.is_at_depot() and (
                    vehicle.battery_inventory - battery_swap - pick_up
                    < vehicle.battery_inventory_capacity * 0.1
                ):
                    # If battery inventory is low, go to depot
                    cluster_id = [
                        depot.id
                        for depot in sorted(
                            self.depots,
                            key=lambda depot: self.get_distance(
                                vehicle.current_location.id, depot.id
                            ),
                        )
                        if depot.get_available_battery_swaps(time)
                        > vehicle.battery_inventory_capacity * 0.9
                    ][0]
                actions.append(
                    Action(
                        swappable_scooters_id[:battery_swap],
                        choose_pick_up(battery_swap, pick_up),
                        [scooter.id for scooter in vehicle.scooter_inventory][
                            :drop_off
                        ],
                        cluster_id,
                    )
                )
        return (
            actions
            if len(actions) > 0
            else [Action([], [], [], neighbour.id) for neighbour in neighbours]
        )

    def do_action(self, action: Action, vehicle: Vehicle, time: int):
        """
        Performs an action on the state -> changing the state + calculates the reward
        :param time: at what time the action is performed
        :param vehicle: Vehicle to perform this action
        :param action: Action - action to be performed on the state
        :return: float - reward for doing the action on the state
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
                vehicle.current_location.add_scooter(delivery_scooter)

        # Moving the state/vehicle from this to next cluster
        vehicle.set_current_location(
            self.get_location_by_id(action.next_location), action
        )

        return refill_time

    def __repr__(self):
        return (
            f"<State: {len(self.get_scooters())} scooters in {len(self.stations)} "
            f"stations with {len(self.vehicles)} vehicles>"
        )

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

    def compute_and_set_trip_intensity(self, sample_scooters):
        clustering.methods.compute_and_set_trip_intensity(self, sample_scooters)

    def sample(self, sample_size: int):
        # Filter out scooters not in sample
        sampled_scooter_ids = random.sample(
            [scooter.id for scooter in self.get_scooters()], sample_size
        )
        for cluster in self.stations:
            cluster.scooters = [
                scooter
                for scooter in cluster.scooters
                if scooter.id in sampled_scooter_ids
            ]

    def get_random_cluster(self, exclude=None):
        return random.choice(
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

    def get_expected_lost_trip_reward(self, lost_trip_reward, exclude=-1):
        # If number of available scooters is less than trip intensity add reward
        return float(
            sum(
                [
                    max(
                        (
                            cluster.trip_intensity_per_iteration
                            - len(cluster.get_available_scooters())
                        ),
                        0,
                    )
                    * lost_trip_reward
                    for cluster in self.stations
                    if cluster.id != exclude
                ]
            )
        )
