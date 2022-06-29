import sim
from sim.LoadSave import LoadSave
import numpy as np
from settings import *
import copy

class State(LoadSave):
    """
    Container class for the whole state of all clusters. Data concerning the interplay between clusters are stored here
    """

    def __init__(
        self,
        stations: [sim.Station] = [],
        depots: [sim.Depot] = [],
        vehicles: [sim.Vehicle] = [],
        scooters_in_use: [sim.Location] = [], # scooters in use (not parked at any station)
        traveltime_matrix=None,
        traveltime_vehicle_matrix=None,
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

        self.locations = self.depots + self.stations
        self.traveltime_matrix = traveltime_matrix
        self.traveltime_vehicle_matrix = traveltime_vehicle_matrix

        if traveltime_matrix is None:
            self.traveltime_matrix = self.calculate_traveltime(SCOOTER_SPEED)

        if traveltime_vehicle_matrix is None:
            self.traveltime_vehicle_matrix = self.calculate_traveltime(VEHICLE_SPEED)

    def sloppycopy(self, *args):
        stationscopy = []
        for s in self.stations:
            stationscopy.append(s.sloppycopy())

        new_state = State(
            stationscopy,
            copy.deepcopy(self.depots),
            copy.deepcopy(self.vehicles),
            copy.deepcopy(self.scooters_in_use),
            traveltime_matrix = self.traveltime_matrix,
            traveltime_vehicle_matrix = self.traveltime_vehicle_matrix,
            rng = self.rng,
        )

        for vehicle in new_state.vehicles:
            vehicle.current_location = new_state.get_location_by_id(
                vehicle.current_location.id
            )

        return new_state

    @staticmethod
    def get_initial_state(bike_class, traveltime_matrix, traveltime_vehicle_matrix, number_of_scooters, number_of_vehicles, leave_intensities, arrive_intensities = None, move_probabilities = None, main_depot = False, secondary_depots = 0, target_state = None, random_seed=None, capacities=None, charging_stations = None, original_ids = None):
        depots = []
        stations = []

        start_of_ids = 0

        num_depots = secondary_depots
        if main_depot:
            num_depots += 1

        if arrive_intensities is None:
            arrive_intensities = leave_intensities

        for station_id in range(len(number_of_scooters)):
            capacity = DEFAULT_STATION_CAPACITY
            if capacities is not None:
                capacity = capacities[station_id]

            tstate = None
            if target_state:
                tstate = target_state[station_id]

            original_id = None
            if original_ids is not None:
                original_id = original_ids[station_id]

            charging_station = None
            if charging_stations is not None:
                charging_station = charging_stations[station_id]

            scooters = []
            for scooter_id in range(number_of_scooters[station_id]):
                if bike_class == "Scooter":
                    scooters.append(sim.Scooter(scooter_id=start_of_ids + scooter_id, battery=100))
                else:
                    scooters.append(sim.Bike(scooter_id=start_of_ids + scooter_id))
            start_of_ids += number_of_scooters[station_id]

            if station_id == 0:
                depots.append(sim.Depot(station_id, main_depot=True, scooters=scooters, leave_intensity_per_iteration=leave_intensities[station_id], arrive_intensity_per_iteration=arrive_intensities[station_id], move_probabilities=move_probabilities[station_id], target_state=tstate, capacity=capacity, original_id=original_id, charging_station=charging_station))

            elif station_id < num_depots:
                depots.append(sim.Depot(station_id, main_depot=False, scooters=scooters, leave_intensity_per_iteration=leave_intensities[station_id], arrive_intensity_per_iteration=arrive_intensities[station_id], move_probabilities=move_probabilities[station_id], target_state=tstate, capacity=capacity, original_id=original_id, charging_station=charging_station))

            else:
                stations.append(sim.Station(station_id, scooters, leave_intensity_per_iteration=leave_intensities[station_id], arrive_intensity_per_iteration=arrive_intensities[station_id], move_probabilities=move_probabilities[station_id], target_state=tstate, capacity=capacity, original_id=original_id, charging_station=charging_station))
                
        state = State(stations, depots, [], traveltime_matrix=traveltime_matrix, traveltime_vehicle_matrix=traveltime_vehicle_matrix)

        state.set_num_vehicles(number_of_vehicles)
        state.set_seed(random_seed)

        return state

    def calculate_traveltime(self, speed):
        traveltime_matrix = []
        for location in self.locations:
            neighbour_traveltime = []
            for neighbour in self.locations:
                neighbour_traveltime.append(location.distance_to(*neighbour.get_location()) / speed)
            traveltime_matrix.append(neighbour_traveltime)
        
        return traveltime_matrix

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_num_vehicles(self, number_of_vehicles):
        self.vehicles = []
        for vehicle_id in range(number_of_vehicles):
            self.vehicles.append(sim.Vehicle(vehicle_id, self.locations[0],
                                             VEHICLE_BATTERY_INVENTORY, VEHICLE_SCOOTER_INVENTORY))

    def set_move_probabilities(self, move_probabilities):
        for st in self.locations:
            st.move_probabilities = move_probabilities[st.id]

    def set_target_state(self, target_state):
        for st in self.locations:
            st.target_state = target_state[st.id]

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
        for cluster in self.locations:
            for scooter in cluster.scooters:
                all_scooters.append(scooter)
        all_scooters.extend(self.scooters_in_use)
        for vehicle in self.vehicles:
            all_scooters.extend(vehicle.scooter_inventory)
            
        return all_scooters

    def get_travel_time(self, start_location_id: int, end_location_id: int):
        return self.traveltime_matrix[start_location_id][end_location_id]

    def get_vehicle_travel_time(self, start_location_id: int, end_location_id: int):
        return self.traveltime_vehicle_matrix[start_location_id][end_location_id]

    def do_action(self, action: sim.Action, vehicle: sim.Vehicle, time: int):
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
            for pick_up_scooter_id in action.pick_ups:
                pick_up_scooter = vehicle.current_location.get_scooter_from_id(
                    pick_up_scooter_id
                )
                # Picking up scooter and adding to vehicle inventory and swapping battery
                vehicle.pick_up(pick_up_scooter)

                # Remove scooter from current cluster
                vehicle.current_location.remove_scooter(pick_up_scooter)
            # Perform all battery swaps
            for battery_swap_scooter_id in action.battery_swaps[:vehicle.battery_inventory]:
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
        for station in self.locations:
            string += f"{str(station)}\n"
        for vehicle in self.vehicles:
            string += f"{str(vehicle)}\n"
        string += f"In use: {self.scooters_in_use}"
        return string

    def get_neighbours(
        self,
        location: sim.Location,
        number_of_neighbours=None,
        is_sorted=True,
        exclude=None,
        not_full=False,
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
            if not_full:
                neighbours = sorted(
                    [
                        state_location
                        for state_location in self.locations
                        if state_location.id != location.id
                        and state_location.id not in (exclude if exclude else [])
                        and state_location.spare_capacity() >= 1
                    ],
                    key=lambda state_location: self.traveltime_matrix[location.id][
                        state_location.id
                    ],
                )
            else:
                neighbours = sorted(
                    [
                        state_location
                        for state_location in self.locations
                        if state_location.id != location.id
                        and state_location.id not in (exclude if exclude else [])
                    ],
                    key=lambda state_location: self.traveltime_matrix[location.id][
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

    def get_vehicle_by_id(self, vehicle_id: int) -> sim.Vehicle:
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
