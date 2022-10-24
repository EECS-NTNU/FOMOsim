import sim
from sim.LoadSave import LoadSave
import numpy as np
from settings import *
import copy
import json
import gzip

class State(LoadSave):
    """
    Container class for the whole state of all stations. Data concerning the interplay between stations are stored here
    """

    def __init__(
        self,
        stations = [],
        vehicles = [],
        bikes_in_use = {}, # bikes not parked at any station
        mapdata=None,
        traveltime_matrix=None,
        traveltime_matrix_stddev=None,
        traveltime_vehicle_matrix=None,
        traveltime_vehicle_matrix_stddev=None,
        rng = None,
    ):
        if rng is None:
            self.rng = np.random.default_rng(None)
        else:
            self.rng = rng

        self.vehicles = vehicles
        self.bikes_in_use = bikes_in_use

        self.set_locations(stations)

        self.traveltime_matrix = traveltime_matrix
        self.traveltime_matrix_stddev = traveltime_matrix_stddev
        self.traveltime_vehicle_matrix = traveltime_vehicle_matrix
        self.traveltime_vehicle_matrix_stddev = traveltime_vehicle_matrix_stddev

        if traveltime_matrix is None:
            self.traveltime_matrix = self.calculate_traveltime(BIKE_SPEED)

        if traveltime_vehicle_matrix is None:
            self.traveltime_vehicle_matrix = self.calculate_traveltime(VEHICLE_SPEED)

        self.mapdata = mapdata

    def sloppycopy(self, *args):
        locationscopy = []
        for s in self.locations:
            locationscopy.append(s.sloppycopy())

        new_state = State(
            locationscopy,
            copy.deepcopy(self.vehicles),
            copy.deepcopy(self.bikes_in_use),
            traveltime_matrix = self.traveltime_matrix,
            traveltime_matrix_stddev = self.traveltime_matrix_stddev,
            traveltime_vehicle_matrix = self.traveltime_vehicle_matrix,
            traveltime_vehicle_matrix_stddev = self.traveltime_vehicle_matrix_stddev,
            rng = self.rng,
        )

        for vehicle in new_state.vehicles:
            vehicle.location = new_state.get_location_by_id(
                vehicle.location.id
            )

        return new_state


    @staticmethod
    def get_initial_state(statedata):
        # create stations

        stations = []

        id_counter = 0

        for station_id, station in enumerate(statedata["stations"]):
            capacity = DEFAULT_STATION_CAPACITY
            if "capacity" in station:
                capacity = station["capacity"]

            original_id = None
            if "original_id" in station:
                original_id = station["original_id"]

            position = None
            if "location" in station:
                position = station["location"]

            charging_station = False
            if "charging_station" in station:
                charging_station = station["charging_station"]

            if ("is_depot" in station) and station["is_depot"]:
                depot_capacity = DEFAULT_DEPOT_CAPACITY
                if "depot_capacity" in station:
                    depot_capacity = station["depot_capacity"]
                stationObj = sim.Depot(station_id, depot_capacity=depot_capacity, capacity=capacity, original_id=original_id, center_location=position, charging_station=charging_station)

            else:
                stationObj = sim.Station(station_id,
                                         capacity=capacity,
                                         original_id=original_id,
                                         center_location=position,
                                         charging_station=charging_station,
                                         leave_intensities = station["leave_intensities"],
                                         leave_intensities_stdev = station["leave_intensities_stdev"],
                                         arrive_intensities = station["arrive_intensities"],
                                         arrive_intensities_stdev = station["arrive_intensities_stdev"],
                                         move_probabilities = station["move_probabilities"],
                                         )

            # create bikes
            bikes = []
            for _ in range(station["num_bikes"]):
                if statedata["bike_class"] == "EBike":
                    bikes.append(sim.EBike(bike_id=id_counter, battery=100))
                else:
                    bikes.append(sim.Bike(bike_id=id_counter))
                id_counter += 1

            stationObj.set_bikes(bikes)

            stations.append(stationObj)

        # create state

        mapdata = None
        if "map" in statedata:
            mapdata = (statedata["map"], statedata["map_boundingbox"])

        state = State(stations,
                      mapdata = mapdata,
                      traveltime_matrix=statedata["traveltime"],
                      traveltime_matrix_stddev=statedata["traveltime_stdev"],
                      traveltime_vehicle_matrix=statedata["traveltime_vehicle"],
                      traveltime_vehicle_matrix_stddev=statedata["traveltime_vehicle_stdev"])

        return state

    def calculate_traveltime(self, speed):
        traveltime_matrix = []
        for location in self.locations:
            neighbour_traveltime = []
            for neighbour in self.locations:
                neighbour_traveltime.append(location.distance_to(*neighbour.get_location()) / speed)
            traveltime_matrix.append(neighbour_traveltime)
        
        return traveltime_matrix

    def set_locations(self, locations):
        self.locations = locations
        self.stations = { station.id : station for station in locations if not isinstance(station, sim.Depot) }
        self.depots = { station.id : station for station in locations if isinstance(station, sim.Depot) }

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_vehicles(self, policies):
        self.vehicles = []
        for vehicle_id, policy in enumerate(policies):
            self.vehicles.append(sim.Vehicle(vehicle_id, self.locations[0], policy, 
                                             VEHICLE_BATTERY_INVENTORY, VEHICLE_BIKE_INVENTORY))

    def set_move_probabilities(self, move_probabilities):
        for st in self.locations:
            st.move_probabilities = move_probabilities[st.id]

    def set_target_state(self, target_state):
        for st in self.locations:
            st.target_state = target_state[st.id]

    def get_station_by_lat_lon(self, lat: float, lon: float):
        """
        :param lat: lat location of bike
        :param lon:
        :return:
        """
        return min(list(self.stations.values()), key=lambda station: station.distance_to(lat, lon))

    def bike_in_use(self, bike):
        self.bikes_in_use[bike.id] = bike

    def remove_used_bike(self, bike):
        del self.bikes_in_use[bike.id]

    def get_used_bike(self):
        if len(self.bikes_in_use) > 0:
            bike = next(iter(self.bikes_in_use))
            remove_used_bike(bike)
            return bike

    # parked bikes
    def get_bikes(self):
        all_bikes = []
        for station in self.stations.values():
            all_bikes.extend(station.get_bikes())
        return all_bikes

    # parked and in-use bikes
    def get_all_bikes(self):
        all_bikes = []
        for station in self.locations:
            all_bikes.extend(station.get_bikes())
        all_bikes.extend(self.bikes_in_use.values())
        for vehicle in self.vehicles:
            all_bikes.extend(vehicle.get_bike_inventory())
            
        return all_bikes

    # parked bikes with usable battery
    def get_num_available_bikes(self):
        num = 0
        for station in self.locations:
            num += len(station.get_available_bikes())
        return num

    def get_travel_time(self, start_location_id: int, end_location_id: int):
        if self.traveltime_matrix_stddev is not None:
            return self.rng.lognormal(self.traveltime_matrix[start_location_id][end_location_id],
                                      self.traveltime_matrix_stddev[start_location_id][end_location_id])
        else:
            return self.traveltime_matrix[start_location_id][end_location_id]

    def get_vehicle_travel_time(self, start_location_id: int, end_location_id: int):
        if self.traveltime_vehicle_matrix_stddev is not None:
            return self.rng.lognormal(self.traveltime_vehicle_matrix[start_location_id][end_location_id],
                                self.traveltime_vehicle_matrix_stddev[start_location_id][end_location_id])
        else:
            return self.traveltime_vehicle_matrix[start_location_id][end_location_id]

    def do_action(self, action, vehicle, time):
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
                vehicle.location.get_available_battery_swaps(time),
            )

            refill_time += vehicle.location.swap_battery_inventory(
                time, batteries_to_swap
            )
            vehicle.add_battery_inventory(batteries_to_swap)

        else:
            for pick_up_bike_id in action.pick_ups:
                pick_up_bike = vehicle.location.get_bike_from_id(
                    pick_up_bike_id
                )
                
                # Picking up bike and adding to vehicle inventory and swapping battery
                vehicle.pick_up(pick_up_bike)

                # Remove bike from current station
                vehicle.location.remove_bike(pick_up_bike)
            # Perform all battery swaps
            for battery_swap_bike_id in action.battery_swaps[:vehicle.battery_inventory]:
                battery_swap_bike = vehicle.location.get_bike_from_id(
                    battery_swap_bike_id
                )
                # Decreasing vehicle battery inventory
                vehicle.change_battery(battery_swap_bike)

            # Dropping of bikes
            for delivery_bike_id in action.delivery_bikes:
                # Removing bike from vehicle inventory
                delivery_bike = vehicle.drop_off(delivery_bike_id)

                # Adding bike to current station and changing coordinates of bike
                vehicle.location.add_bike(self.rng, delivery_bike)

        # Moving the state/vehicle from this to next station
        vehicle.location = self.get_location_by_id(action.next_location)

        return refill_time

    def __repr__(self):
        string = f"<State: {len(self.get_bikes())} bikes in {len(self.stations)} stations with {len(self.vehicles)} vehicles>\n"
        for station in self.locations:
            string += f"{str(station)}\n"
        for vehicle in self.vehicles:
            string += f"{str(vehicle)}\n"
        string += f"In use: {len(self.bikes_in_use.values())}"
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
        Get sorted list of stations closest to input station
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
        return self.locations[location_id]

    def sample(self, sample_size: int):
        # Filter out bikes not in sample
        sampled_bike_ids = self.rng.choice(
            [bike.id for bike in self.get_bikes()], sample_size, replace=False,
        )
        for station in self.stations.values():
            station.set_bikes([
                bike
                for bike in station.get_bikes()
                if bike.id in sampled_bike_ids
            ])

    def get_vehicle_by_id(self, vehicle_id):
        """
        Returns the vehicle object in the state corresponding to the vehicle id input
        :param vehicle_id: the id of the vehicle to fetch
        :return: vehicle object
        """
        return self.vehicles[vehicle_id]
