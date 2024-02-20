import sim
from sim.LoadSave import LoadSave
import numpy as np
from settings import *
import copy
import json
import gzip
import os
# from policies.inngjerdingen_moeller.parameters_MILP import MILP_data 


class State(LoadSave):
    """
    Container class for the whole state of all stations. Data concerning the interplay between stations are stored here
    """

    def __init__(
        self,
        locations = {},
        stations = {},
        areas = {},
        vehicles = {},
        bikes_in_use = {}, # bikes not parked at any station
        mapdata=None,
        traveltime_matrix=None,
        traveltime_matrix_stddev=None,
        traveltime_vehicle_matrix=None,
        traveltime_vehicle_matrix_stddev=None, 
        rng = None,
        rng2 = None,
        seed = None
    ):
        if rng is None:
            self.rng = np.random.default_rng(None)
        else:
            self.rng = rng

        if rng2 is None:
            self.rng2 = np.random.default_rng(None)
        else:
            self.rng2 = rng2

        self.vehicles = vehicles
        self.bikes_in_use = bikes_in_use
        self.seed = seed

        self.set_locations(locations)


        self.traveltime_matrix = traveltime_matrix
        self.traveltime_matrix_stddev = traveltime_matrix_stddev
        self.traveltime_vehicle_matrix = traveltime_vehicle_matrix
        self.traveltime_vehicle_matrix_stddev = traveltime_vehicle_matrix_stddev

        if traveltime_matrix is None:
            self.traveltime_matrix = self.calculate_traveltime(BIKE_SPEED)

        if traveltime_vehicle_matrix is None:
            self.traveltime_vehicle_matrix = self.calculate_traveltime(VEHICLE_SPEED)

        self.mapdata = mapdata

    # TODO
    def sloppycopy(self, *args):
        locationscopy = {}
        for id, loc in self.locations.items():
            locationscopy[id] = loc

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
                vehicle.location.location_id
            )

        return new_state

    @staticmethod
    def get_initial_state(sb_statedata = None, ff_statedata = None):
        # create stations
        sb_state = State.get_initial_sb_state(sb_statedata) if sb_statedata else None
        ff_state = State.get_initial_ff_state(ff_statedata) if ff_statedata else None

        # Add sb locations to ff state, and thus all the bikes
        merged_state = ff_state.sloppycopy()
        for location_id, location in sb_state.locations.keys():
            if location_id not in ff_state.get_location_ids(): #Should not be a problem since all stations are marked with S and areas are marked with A. Might have to change for Depots
                merged_state.locations[location_id] = location
        merged_state.set_locations(merged_state.locations.values()) # Update all dictionaries with locations (locations, stations, areas)

        merged_state.vehicles.extend(sb_state.vehicles)

    @staticmethod
    def get_initial_ff_state(statedata):
        # create areas
        areas = []

        id_counter = 0
        for area_id, area in enumerate(statedata["areas"]):
            capacity = float('inf') # TODO trenger vi denne?

            center_position = None
            if "center_location" in area:
                center_position = area["center_location"] # (lat, lon)
            
            border_vertices = None
            if "border_vertices" in area:
                border_vertices = area["border_vertices"] # [(lat, lan), x7]

            areaObj = sim.Area("A" + str(area_id),
                               border_vertices,
                               center_location = center_position,
                               capacity = capacity,
                               leave_intensities = area["leave_intensities"],
                               leave_intensities_stdev = area["leave_intensities_stdev"],
                               arrive_intensities = area["arrive_intensities"],
                               arrive_intensities_stdev = area["arrive_intensities_stdev"],
                               move_probabilities = area["move_probabilities"]
                               )

            bikes = {}
            for battery in area["bikes"]:
                bikes.append(sim.EScooter(bike_id = "ES"+str(id_counter), battery_level = battery))
                id_counter += 1

            areaObj.set_bikes(bikes)

            areas.append(areaObj)

        # TODO - hvordan skal disse se ut?
        for depot_id, depot in enumerate(statedata["depots"]):
            pass

        # TODO - dette må sikkert fikses
        mapdata = None
        if "map" in statedata:
            mapdata = (statedata["map"], statedata["map_boundingbox"])

        # TODO Skal vi ha travel_matrix?
        state = State(areas = areas,
                      mapdata = mapdata,
                      traveltime_matrix=statedata["traveltime"],
                      traveltime_matrix_stddev=statedata["traveltime_stdev"],
                      traveltime_vehicle_matrix=statedata["traveltime_vehicle"],
                      traveltime_vehicle_matrix_stddev=statedata["traveltime_vehicle_stdev"])
        
        # Må nok testes
        vertex_to_area = {}
        for area_id, area in areas.items():
            for vertex in area.border_vertices:
                if vertex not in vertex_to_area:
                    vertex_to_area[vertex] = [area]
                else:
                    vertex_to_area[vertex].append(area)

        for area_id, area in areas.items():
            neighbors = set()
            for vertex in area.border_vertices:
                neighbors.update(vertex_to_area[vertex])
            neighbors.discard(area)
            area.neighboring_areas = list(neighbors)

        return state

    @staticmethod
    def get_initial_sb_state(statedata):

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

            # TODO fix depot init
            if ("is_depot" in station) and station["is_depot"]:
                depot_capacity = DEFAULT_DEPOT_CAPACITY
                if "depot_capacity" in station:
                    depot_capacity = station["depot_capacity"]
                stationObj = sim.Depot("S" + str(station_id),
                                         capacity=capacity,
                                         depot_capacity= depot_capacity,
                                         original_id=original_id,
                                         center_location=position,
                                         charging_station=charging_station,
                                         leave_intensities = station["leave_intensities"],
                                         leave_intensities_stdev = station["leave_intensities_stdev"],
                                         arrive_intensities = station["arrive_intensities"],
                                         arrive_intensities_stdev = station["arrive_intensities_stdev"],
                                         move_probabilities = station["move_probabilities"],
                                         )
            else:
                stationObj = sim.Station("S" + str(station_id),
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
                    bikes.append(sim.EBike(bike_id= "EB" + str(id_counter), battery=100)) # TODO legge til funksjonalitet for at start batteri nivå er satt i json
                else:
                    bikes.append(sim.Bike(bike_id= "B"+str(id_counter)))
                id_counter += 1

            stationObj.set_bikes(bikes)

            stations.append(stationObj)

        # create state
        mapdata = None
        if "map" in statedata:
            mapdata = (statedata["map"], statedata["map_boundingbox"])

        traveltime_matrix = {} if statedata["traveltime"] else None
        traveltime_matrix_stddev = {} if statedata["traveltime_stdev"] else None
        traveltime_vehicle_matrix = {} if statedata["traveltime_vehicle"] else None
        traveltime_vehicle_matrix_stddev = {} if statedata["traveltime_vehicle_stdev"] else None
        for i in range(len(statedata["traveltime"])):
            for y in range(len(statedata["traveltime"])):
                if statedata["traveltime"]:
                    traveltime_matrix[("S"+str(i), "S"+str(y))] = statedata["traveltime"][i][y]
                if statedata["traveltime_stdev"]:
                    traveltime_matrix_stddev[("S"+str(i), "S"+str(y))] = statedata["traveltime_stdev"][i][y]
                if statedata["traveltime_vehicle"]:
                    traveltime_vehicle_matrix[("S"+str(i), "S"+str(y))] = statedata["traveltime_vehicle"][i][y]
                if statedata["traveltime_vehicle_stdev"]:
                    traveltime_vehicle_matrix_stddev[("S"+str(i), "S"+str(y))] = statedata["traveltime_vehicle_stdev"][i][y]

        state = State(locations= stations,
                      mapdata = mapdata,
                      traveltime_matrix=traveltime_matrix,
                      traveltime_matrix_stddev=traveltime_matrix_stddev,
                      traveltime_vehicle_matrix=traveltime_vehicle_matrix,
                      traveltime_vehicle_matrix_stddev=traveltime_vehicle_matrix_stddev)
        
        neighbor_dict = state.read_neighboring_stations_from_file()
        for station in stations:
            station.set_neighboring_stations(neighbor_dict, state.locations)

        return state

    def calculate_traveltime(self, speed):
        traveltime_matrix = {}
        for location in self.get_locations():
            for neighbour in self.get_locations():
                traveltime_matrix[(location.location_id, neighbour.location_id)] = (location.distance_to(*neighbour.get_location()) / speed)*60 #multiplied by 60 to get minutes
        return traveltime_matrix

    def set_locations(self, locations):
        self.locations = {location.location_id: location for location in locations}
        self.stations = { station.location_id : station for station in locations if isinstance(station, sim.Station)}# and not isinstance(station, sim.Depot)}
        self.depots = { station.location_id : station for station in locations if isinstance(station, sim.Depot) }
        self.areas = { area.location_id : area for area in locations if isinstance(area, sim.Area) }

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def set_vehicles(self, policies):
        self.vehicles = []
        for vehicle_id, policy in enumerate(policies):
            self.vehicles.append(sim.Vehicle(vehicle_id, self.locations["S0"], policy, VEHICLE_BATTERY_INVENTORY, VEHICLE_BIKE_INVENTORY))

    def set_move_probabilities(self, move_probabilities):
        for st in self.get_locations():
            st.move_probabilities = move_probabilities[st.location_id]

    def set_target_state(self, target_state):
        for st in self.get_locations():
            st.target_state = target_state[st.location_id]

    def get_station_by_lat_lon(self, lat: float, lon: float):
        """
        :param lat: lat location of bike
        :param lon:
        :return:
        """
        return min(list(self.stations.values()), key=lambda station: station.distance_to(lat, lon))

    def bike_in_use(self, bike):
        self.bikes_in_use[bike.bike_id] = bike

    def remove_used_bike(self, bike):
        del self.bikes_in_use[bike.bike_id]

    # TODO hva betyr denne?
    def get_used_bike(self):
        if len(self.bikes_in_use) > 0:
            bike = next(iter(self.bikes_in_use))
            self.remove_used_bike(bike)
            return bike

    # parked bikes
    def get_bikes(self):
        all_bikes = []
        for station in self.get_locations():
            all_bikes.extend(station.get_bikes())
        return all_bikes

    # parked and in-use bikes
    def get_all_bikes(self):
        all_bikes = []
        for station in self.get_locations():
            all_bikes.extend(station.get_bikes())
        all_bikes.extend(self.bikes_in_use.values())
        for vehicle in self.vehicles:
            all_bikes.extend(vehicle.get_bike_inventory())
            
        return all_bikes

    # parked bikes with usable battery
    def get_num_available_bikes(self):
        num = 0
        for station in self.get_stations():
            num += len(station.get_available_bikes())
        return num

    # parked escooters with usable battery
    def get_num_available_escooters(self):
        num = 0
        for area in self.get_areas():
            num += len(area.get_available_bikes())
        return num

    def get_travel_time(self, start_location_id: int, end_location_id: int):
        if self.traveltime_matrix_stddev is not None:
            return self.rng2.lognormal(self.traveltime_matrix[(start_location_id, end_location_id)],
                                      self.traveltime_matrix_stddev[(start_location_id, end_location_id)])
        else:
            return self.traveltime_matrix[(start_location_id, end_location_id)]

    def get_vehicle_travel_time(self, start_location_id: int, end_location_id: int):
        if self.traveltime_vehicle_matrix_stddev is not None:
            return self.rng2.lognormal(self.traveltime_vehicle_matrix[(start_location_id, end_location_id)],
                                self.traveltime_vehicle_matrix_stddev[(start_location_id, end_location_id)])
        else:
            return self.traveltime_vehicle_matrix[(start_location_id, end_location_id)]

    # TODO
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
                vehicle.get_flat_batteries(),
                vehicle.location.get_available_battery_swaps(time),
            )

            refill_time += vehicle.location.swap_battery_inventory(
                time, batteries_to_swap
            )
            vehicle.add_battery_inventory(batteries_to_swap)

            #### TODO - legg til tid for dette
            for e_scooter in vehicle.get_bike_inventory():
                e_scooter.swap_battery()
                
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
            for battery_swap_bike_id in action.battery_swaps:
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
                vehicle.location.add_bike(delivery_bike)

        # Moving the state/vehicle from this to next station
        vehicle.location = self.get_location_by_id(action.next_location)

        return refill_time

    def __repr__(self):
        string = f"<State: {len(self.get_bikes())} bikes in {len(self.stations)} stations with {len(self.vehicles)} vehicles>\n"
        for station in self.get_locations():
            string += f"{str(station)}\n"
        for vehicle in self.get_locations():
            string += f"{str(vehicle)}\n"
        string += f"In use: {len(self.bikes_in_use.values())}"
        return string

    # TODO
    def get_neighbours(
        self,
        location: sim.Location,
        number_of_neighbours=None,
        is_sorted=True,
        exclude=None,
        not_full=False,
        not_empty=False
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
            for state_location in self.get_locations()
            if state_location.location_id != location.location_id
            and state_location.location_id not in (exclude if exclude else [])
        ]
        if is_sorted:
            if not_full:
                neighbours = sorted(
                    [
                        state_location
                        for state_location in self.get_locations()
                        if state_location.location_id != location.location_id
                        and state_location.location_id not in (exclude if exclude else [])
                        and state_location.spare_capacity() >= 1
                    ],
                    key=lambda state_location: self.traveltime_matrix[(location.location_id, state_location.location_id)],
                )
            elif not_empty:
                neighbours = sorted(
                    [
                        state_location
                        for state_location in self.get_locations()
                        if state_location.location_id != location.location_id
                        and state_location.location_id not in (exclude if exclude else [])
                        and len(state_location.get_available_bikes()) >= 1
                    ],
                    key=lambda state_location: self.traveltime_matrix[(location.location_id, state_location.location_id)],
                )
            else:
                neighbours = sorted(
                    [
                        state_location
                        for state_location in self.get_locations()
                        if state_location.location_id != location.location_id
                        and state_location.location_id not in (exclude if exclude else [])
                    ],
                    key=lambda state_location: self.traveltime_matrix[(location.location_id, state_location.location_id)],
                )

        test = neighbours[:number_of_neighbours] if number_of_neighbours else neighbours
        if len(test) == 0:
            print('number of neighmors', neighbours[:number_of_neighbours], number_of_neighbours)
            print('neighbors', neighbours)
                
        return neighbours[:number_of_neighbours] if number_of_neighbours else neighbours

    def get_location_by_id(self, location_id):
        return self.locations[location_id]
    
    def get_location_ids(self):
        return list(self.locations.keys())
    
    def get_locations(self):
        return list(self.locations.values())
    
    def get_area_by_id(self, area_id):
        return self.areas[area_id]
    
    def get_area_ids(self):
        return list(self.areas.keys())
    
    def get_areas(self):
        return list(self.areas.values())
    
    def get_station_by_id(self, station_id):
        return self.locations[station_id]
    
    def get_station_ids(self):
        return list(self.stations.keys())
    
    def get_stations(self):
        return list(self.stations.values())
    
    # TODO forstå denne
    def sample(self, sample_size: int):
        # Filter out bikes not in sample
        sampled_bike_ids = self.rng2.choice(
            [bike.bike_id for bike in self.get_bikes()], sample_size, replace=False,
        )
        for station in self.stations.values():
            station.set_bikes([
                bike
                for bike in station.get_bikes()
                if bike.bike_id in sampled_bike_ids
            ])

    def get_vehicle_by_id(self, vehicle_id):
        """
        Returns the vehicle object in the state corresponding to the vehicle id input
        :param vehicle_id: the id of the vehicle to fetch
        :return: vehicle object
        """
        return self.vehicles[vehicle_id]
    
    # TODO Trenger vi denne, evt. hva gjør vi for FF?
    def read_neighboring_stations_from_file(self):
        neighboring_stations = dict()      #{station_ID: [list of station_IDs]}
        filename = 'policies/inngjerdingen_moeller/saved_time_data/' + (self.mapdata[0]).split('.')[0].split('/')[-1] +'_static_data.json'
        if not os.path.exists(filename):
            print("JSON-file", filename, "does not exist")

        with open(filename,'r') as infile:
            json_dict = json.load(infile)
        for key, value in json_dict["neighboring_stations"].items():
            neighboring_stations[int(key)] = value


        return neighboring_stations
