import sim
from sim.LoadSave import LoadSave
import numpy as np
from settings import *
import copy
import json
import gzip
import os
import random
import geopy
# from policies.inngjerdingen_moeller.parameters_MILP import MILP_data 


class State(LoadSave):
    """
    Container class for the whole state of all stations. Data concerning the interplay between stations are stored here
    """

    def __init__(
        self,
        locations = {},
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

        # if traveltime_matrix is None:
        #     self.traveltime_matrix = self.calculate_traveltime(BIKE_SPEED)

        # if traveltime_vehicle_matrix is None:
        #     self.traveltime_vehicle_matrix = self.calculate_traveltime(VEHICLE_SPEED)

        # if traveltime_matrix_stddev is None:
        #     self.traveltime_matrix_stddev = {key: 0 for key in self.traveltime_matrix}
        
        # if traveltime_vehicle_matrix_stddev is None:
        #     self.traveltime_vehicle_matrix_stddev = {key: 0 for key in self.traveltime_vehicle_matrix}

        self.mapdata = mapdata

    # TODO
    def sloppycopy(self, *args):
        new_state = State(
            list(self.get_locations()),
            copy.deepcopy(self.vehicles),
            copy.deepcopy(self.bikes_in_use),
            traveltime_matrix = self.traveltime_matrix,
            traveltime_matrix_stddev = self.traveltime_matrix_stddev,
            traveltime_vehicle_matrix = self.traveltime_vehicle_matrix,
            traveltime_vehicle_matrix_stddev = self.traveltime_vehicle_matrix_stddev,
            rng = self.rng,
        )

        for vehicle in new_state.get_vehicles():
            vehicle.location = new_state.get_location_by_id(
                vehicle.location.location_id
            )

        return new_state

    @staticmethod
    def get_initial_state(sb_statedata = None, ff_statedata = None):
        # create stations
        sb_state = State.get_initial_sb_state(sb_statedata) if sb_statedata else None
        sb_depots = len(sb_state.depots) if sb_state else 0
        ff_state = State.get_initial_ff_state(ff_statedata, num_depots = sb_depots) if ff_statedata else None

        if sb_state and not ff_state: # if only ff_state = None
            return sb_state
        elif ff_state and not sb_state: # if only sb_state = None
            return ff_state

        # Add sb locations to ff state, and thus all the bikes
        merged_state = ff_state.sloppycopy()
        for location_id, location in sb_state.locations.items():
            if location_id not in ff_state.get_location_ids(): #Should not be a problem since all stations are marked with S and areas are marked with A. Might have to change for Depots
                merged_state.locations[location_id] = location
        merged_state.set_locations(merged_state.locations.values()) # Update all dictionaries with locations (locations, stations, areas)
        
        for station in merged_state.get_stations():
            area = merged_state.get_area_by_lat_lon(station.get_lat(), station.get_lon())
            station.area = area.location_id
            area.station = station.location_id

        merged_state.mapdata = sb_state.mapdata
        
        merged_state.traveltime_matrix = State.__merge_dicts(sb_state.traveltime_matrix, ff_state.traveltime_matrix)
        merged_state.traveltime_matrix_stddev = State.__merge_dicts(sb_state.traveltime_matrix_stddev, ff_state.traveltime_matrix_stddev)
        merged_state.traveltime_vehicle_matrix = State.__merge_dicts(sb_state.traveltime_vehicle_matrix, ff_state.traveltime_vehicle_matrix)
        merged_state.traveltime_vehicle_matrix_stddev = State.__merge_dicts(sb_state.traveltime_vehicle_matrix_stddev, ff_state.traveltime_vehicle_matrix_stddev)

        return merged_state

    @staticmethod
    def __merge_dicts(dict1, dict2):
        # Check if both dictionaries are None
        if dict1 is None and dict2 is None:
            return None
        # Check if one of the dictionaries is None, then return the other
        elif dict1 is None:
            return dict2
        elif dict2 is None:
            return dict1
        else:
            # Merge both dictionaries and return the result
            merged_dict = {**dict1, **dict2}
            return merged_dict

    @staticmethod
    def get_initial_ff_state(statedata, num_depots = 0):
        # create areas
        areas = []

        escooter_id_counter = 0
        area_num = 0 # TODO denne trengs ikke alltid
        for area in statedata["areas"]:

            area_id = "A" + str(area_num)
            if "id" in area:
                area_id = area["id"]

            center_position = None
            if "location" in area:
                center_position = area["location"]
                
                lat, lon = center_position[0], center_position[1]
                min_lat, max_lat, min_lon, max_lon = 63.415774, 63.440144, 10.370453, 10.431571
                if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                    continue
            
            border_vertices = None
            if "edges" in area:
                edge_list = area["edges"]
                border_vertices = [(edge[0], edge[1]) for edge in edge_list]

            leave_intensities = None
            if "depature_intensities" in area:
                leave_intensities = area["depature_intensities"]
            else:
                leave_intensities = [[np.random.random()*2 for _ in range(24)] for _ in range(7)]

            arrive_intensities = None
            if "arrival_intensities" in area:
                arrive_intensities = area["arrival_intensities"] # [[lat, lan], x7]
            else:
                arrive_intensities = [[np.random.random()*2 for _ in range(24)] for _ in range(7)]

            move_probabilities = None
            if "move_probabilites" in area:
                move_probabilities = area["move_probabilites"]

            leave_intensities_stdev = None
            if "depature_intensities_stdev" in area:
                leave_intensities_stdev = area["depature_intensities_stdev"]
            else:
                leave_intensities_stdev = [[0 for _ in range(24)] for _ in range(7)]

            arrival_intensities_stdev = None
            if "arrival_intensities_stdev" in area:
                arrival_intensities_stdev = area["arrival_intensities_stdev"]
            else:
                arrival_intensities_stdev = [[0 for _ in range(24)] for _ in range(7)]

            areaObj = sim.Area(area_id,
                               border_vertices,
                               center_location = center_position,
                               leave_intensities = leave_intensities,
                               leave_intensities_stdev = leave_intensities_stdev,
                               arrive_intensities = arrive_intensities,
                               arrive_intensities_stdev = arrival_intensities_stdev,
                               move_probabilities = move_probabilities
                               )

            bikes = []
            for battery in area["e_scooters"]:
                bikes.append(sim.EScooter(*(center_position), location_id = areaObj.location_id, bike_id = "ES"+str(escooter_id_counter), battery = battery))
                escooter_id_counter += 1

            areaObj.set_bikes(bikes)

            areas.append(areaObj)

            area_num += 1

        # TODO - hvordan skal disse se ut?
        if 'depots' in statedata:
            for depot in statedata['depots']:
                depotObj = sim.Depot(depot['id'], False, center_location= depot['location'])
                areas.append(depotObj)

        # TODO - dette må sikkert fikses
        mapdata = None
        if "map" in statedata:
            mapdata = (statedata["map"], statedata["map_boundingbox"])
            
        traveltime_matrix = None
        if "traveltime_matrix" in statedata:
            traveltime_matrix = {tuple(map(str, key.split("__"))): value for key, value in statedata["traveltime_matrix"].items()}

        traveltime_vehicle_matrix = None
        if "traveltime_vehicle_matrix" in statedata:
            traveltime_vehicle_matrix = {tuple(map(str, key.split("__"))): value for key, value in statedata["traveltime_vehicle_matrix"].items()}


        state = State(locations = areas,
                      mapdata = mapdata,
                      traveltime_matrix=traveltime_matrix,
                    #   traveltime_matrix_stddev=statedata["traveltime_stdev"],
                      traveltime_vehicle_matrix=traveltime_vehicle_matrix,
                    #   traveltime_vehicle_matrix_stddev=statedata["traveltime_vehicle_stdev"]
                      )
        
        # Må nok testes
        vertex_to_area = {}
        for area_id, area in state.areas.items():
            for vertex in area.border_vertices:
                if vertex not in vertex_to_area.keys():
                    vertex_to_area[vertex] = [area]
                else:
                    vertex_to_area[vertex].append(area)

        for area_id, area in state.areas.items():
            neighbors = set()
            for vertex in area.border_vertices:
                neighbors.update(vertex_to_area[vertex])
            neighbors.discard(area)
            area.neighbours = list(neighbors)

        # for _ in range(400):
        #     area = random.choice(state.get_areas())
        #     area.add_bike(sim.EScooter(area.lat, area.lon, area.location_id, f'ES{escooter_id_counter}', random.randint(30,100)))
        #     escooter_id_counter += 1

        return state

    @staticmethod
    def get_initial_sb_state(statedata):
        locations = []
        num_stations = 0
        num_depots = 0

        num_ebikes = 0
        num_bikes = 0
        for station in statedata["stations"]:
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
                stationObj = sim.Depot("D" + str(num_depots),
                                         is_station_based = True,
                                         capacity=capacity,
                                         depot_capacity= depot_capacity,
                                         original_id=original_id,
                                         center_location=position,
                                         )
            else:
                stationObj = sim.Station("S" + str(num_stations),
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
                if statedata["bike_class"] == "EBike": # TODO har et system enten kun bare vanlig sykler eller el-sykler?
                    bikes.append(sim.EBike(bike_id= "EB" + str(num_ebikes), battery=100)) # TODO legge til funksjonalitet for at start batteri nivå er satt i json
                    num_ebikes += 1
                else:
                    bikes.append(sim.Bike(bike_id= "B"+str(num_bikes), is_station_based=True))
                    num_bikes += 1

            stationObj.set_bikes(bikes)

            if isinstance(stationObj, sim.Depot):
                num_depots += 1
            else:
                num_stations += 1
            locations.append(stationObj)

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
                    traveltime_matrix[(locations[i].location_id, locations[y].location_id)] = statedata["traveltime"][i][y]
                if statedata["traveltime_stdev"]:
                    traveltime_matrix_stddev[(locations[i].location_id, locations[y].location_id)] = statedata["traveltime_stdev"][i][y]
                if statedata["traveltime_vehicle"]:
                    traveltime_vehicle_matrix[(locations[i].location_id, locations[y].location_id)] = statedata["traveltime_vehicle"][i][y]
                if statedata["traveltime_vehicle_stdev"]:
                    traveltime_vehicle_matrix_stddev[(locations[i].location_id, locations[y].location_id)] = statedata["traveltime_vehicle_stdev"][i][y]

        state = State(locations= locations,
                      mapdata = mapdata,
                      traveltime_matrix=traveltime_matrix,
                      traveltime_matrix_stddev=traveltime_matrix_stddev,
                      traveltime_vehicle_matrix=traveltime_vehicle_matrix,
                      traveltime_vehicle_matrix_stddev=traveltime_vehicle_matrix_stddev)
        
        neighbor_dict = state.read_neighboring_stations_from_file()
        for station in [location for location in locations if isinstance(location, sim.Station) and not isinstance(location, sim.Depot)]:
            station.set_neighboring_stations(neighbor_dict, locations)
            station.set_move_probabilities(locations)
        return state

    def calculate_traveltime(self, speed):
        locations = [(loc, loc.get_location()) for loc in self.get_locations()]
        traveltime_matrix = {}

        for i, (location, loc_coords) in enumerate(locations):
            for neighbour, neighbour_coords in locations[i+1:]:  # Start from the next location to avoid duplicates
                distance = location.distance_to(*neighbour_coords)
                travel_time = (distance / speed) * 60  # Calculate travel time once
                # Set travel time for both directions due to symmetry
                traveltime_matrix[(location.location_id, neighbour.location_id)] = travel_time
                traveltime_matrix[(neighbour.location_id, location.location_id)] = travel_time
        return traveltime_matrix

        # traveltime_matrix = {}
        # for location in self.get_locations():
        #     for neighbour in self.get_locations():
        #         traveltime_matrix[(location.location_id, neighbour.location_id)] = (location.distance_to(*neighbour.get_location()) / speed)*60 #multiplied by 60 to get minutes
        # return traveltime_matrix

    def set_locations(self, locations):
        self.locations = {location.location_id: location for location in locations}
        self.stations = { station.location_id : station for station in locations if isinstance(station, sim.Station) and not isinstance(station, sim.Depot)}
        self.depots = { station.location_id : station for station in locations if isinstance(station, sim.Depot) }
        self.areas = { area.location_id : area for area in locations if isinstance(area, sim.Area) }

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def set_vehicles(self, policies):
        for policy in policies:
            num_vehicles = len(self.vehicles)
            self.vehicles["V" + str(num_vehicles)] = sim.Vehicle("V" + str(num_vehicles), 
                                             start_location = self.locations["S0"], 
                                             policy = policy, 
                                             battery_inventory_capacity = VEHICLE_BATTERY_INVENTORY, 
                                             bike_inventory_capacity = VEHICLE_BIKE_INVENTORY, 
                                             is_station_based = None)

    def set_sb_vehicles(self, policies):
        for policy in policies:
            num_vehicles = len(self.vehicles)
            self.vehicles["V" + str(num_vehicles)] = sim.Vehicle("V" + str(num_vehicles), 
                                             start_location = self.locations["S0"], 
                                             policy = policy, 
                                             battery_inventory_capacity = VEHICLE_BATTERY_INVENTORY, 
                                             bike_inventory_capacity = VEHICLE_BIKE_INVENTORY, 
                                             is_station_based = True)

    def set_ff_vehicles(self, policies):
        for policy in policies:
            num_vehicles = len(self.vehicles)
            self.vehicles["V" + str(num_vehicles)] = sim.Vehicle("V" + str(num_vehicles), 
                                             start_location = self.locations["A0"], 
                                             policy = policy, 
                                             battery_inventory_capacity = VEHICLE_BATTERY_INVENTORY, 
                                             bike_inventory_capacity = VEHICLE_BIKE_INVENTORY, 
                                             is_station_based = False)
            
    def set_move_probabilities(self, move_probabilities):
        for st in self.get_locations():
            st.move_probabilities = move_probabilities[st.location_id]

    def set_target_state(self, target_state):
        for st in self.get_locations():
            st.target_state = target_state[st.location_id]

    def get_station_by_lat_lon(self, lat: float, lon: float):
        """
        Returns a list of Station-objects in acsending order by distance to given lat and lon.
        """
        return min(list(self.stations.values()), key=lambda station: station.distance_to(lat, lon))
    
    def get_area_by_lat_lon(self, lat: float, lon: float):
        return min(list(self.get_areas()), key=lambda area: area.distance_to(lat, lon))

    def set_bike_in_use(self, bike):
        self.bikes_in_use[bike.bike_id] = bike

    def remove_used_bike(self, bike):
        del self.bikes_in_use[bike.bike_id]

    def get_sb_bikes_in_use(self):
        return [bike for bike in self.bikes_in_use.values() if bike.is_station_based]
    
    def get_ff_bikes_in_use(self):
        return [bike for bike in self.bikes_in_use.values() if not bike.is_station_based]

    # Not used when FULL_TRIP = True
    def get_used_bike(self):
        if len(self.bikes_in_use) > 0:
            bike = next(iter(self.bikes_in_use))
            self.remove_used_bike(bike)
            return bike

    def get_parked_bikes(self):
        parked_bikes = []
        for location in self.get_locations():
            parked_bikes.extend(location.get_bikes())
        return parked_bikes
    
    def get_parked_sb_bikes(self):
        parked_sb_bikes = []
        for station in self.get_stations():
            parked_sb_bikes.extend(station.get_bikes())
        return parked_sb_bikes
    
    def get_parked_ff_bikes(self):
        parked_ff_bikes = []
        for area in self.get_areas():
            parked_ff_bikes.extend(area.get_bikes())
        return parked_ff_bikes

    # parked and in-use bikes
    def get_all_bikes(self):
        all_bikes = self.get_parked_bikes()
        all_bikes.extend(self.bikes_in_use.values())
        for vehicle in self.get_vehicles():
            all_bikes.extend(vehicle.get_bike_inventory())

        return all_bikes
    
    def get_all_sb_bikes(self):
        all_bikes = self.get_parked_sb_bikes()
        all_bikes.extend(self.get_sb_bikes_in_use())
        for vehicle in self.get_vehicles():
            all_bikes.extend(vehicle.get_sb_bike_inventory())

        return all_bikes
    
    def get_all_ff_bikes(self):
        all_bikes = self.get_parked_ff_bikes()
        all_bikes.extend(self.get_ff_bikes_in_use())
        for vehicle in self.get_vehicles():
            all_bikes.extend(vehicle.get_ff_bike_inventory())

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

    def get_travel_time(self, start_location_id, end_location_id):
        start_loc = self.get_location_by_id(start_location_id)
        end_loc = self.get_location_by_id(end_location_id)
        if isinstance(start_loc, sim.Area) or isinstance(end_loc, sim.Area) or isinstance(start_loc, sim.Depot) or isinstance(end_loc, sim.Depot):
            return (geopy.distance.distance((start_loc.lat, start_loc.lon), (end_loc.lat, end_loc.lon)).km / BIKE_SPEED) * 60

        elif self.traveltime_matrix_stddev is not None and len(self.traveltime_matrix_stddev) == len(self.traveltime_matrix): # and (start_location_id, end_location_id) in self.traveltime_matrix_stddev:
            return self.rng2.lognormal(self.traveltime_matrix[(start_location_id, end_location_id)], 
                                       self.traveltime_matrix_stddev[(start_location_id, end_location_id)])
        return self.traveltime_matrix[(start_location_id, end_location_id)]

    def get_vehicle_travel_time(self, start_location_id, end_location_id):
        start_loc = self.get_location_by_id(start_location_id)
        end_loc = self.get_location_by_id(end_location_id)
        if isinstance(start_loc, sim.Area) or isinstance(end_loc, sim.Area) or isinstance(start_loc, sim.Depot) or isinstance(end_loc, sim.Depot):
            return (geopy.distance.distance((start_loc.lat, start_loc.lon), (end_loc.lat, end_loc.lon)).km / VEHICLE_SPEED) * 60

        elif self.traveltime_vehicle_matrix_stddev is not None and len(self.traveltime_vehicle_matrix_stddev) == len(self.traveltime_vehicle_matrix): # and (start_location_id, end_location_id) in self.traveltime_vehicle_matrix_stddev:
            if (start_location_id, end_location_id) in self.traveltime_vehicle_matrix_stddev.keys():
                return self.rng2.lognormal(self.traveltime_vehicle_matrix[(start_location_id, end_location_id)],
                                           self.traveltime_vehicle_matrix_stddev[(start_location_id, end_location_id)])
        return self.traveltime_vehicle_matrix[(start_location_id, end_location_id)]

    # TODO remove debugger
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
            if vehicle.cluster is None: # TODO hvis du er i en stasjon
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

                for helping_pickup_id in action.helping_pickup:
                    helping_pickup_bike = action.helping_cluster.get_bike_from_id(
                        helping_pickup_id
                    )

                    self.get_location_by_id(helping_pickup_bike.location_id).remove_bike(helping_pickup_bike)
                    
                    # Picking up bike and adding to vehicle inventory and swapping battery
                    vehicle.pick_up(helping_pickup_bike)

                for helping_delivery_id in action.helping_delivery:
                    helping_delivery_bike = vehicle.drop_off(helping_delivery_id)

                    self.get_location_by_id(vehicle.location.area).add_bike(helping_delivery_bike)

            else:
                for pick_up_bike_id in action.pick_ups:
                    pick_up_bike = vehicle.cluster.get_bike_from_id(pick_up_bike_id)

                    # Remove bike from current station
                    current_location = self.get_location_by_id(pick_up_bike.location_id)
                    current_location.remove_bike(pick_up_bike)
                    
                    # Picking up bike and adding to vehicle inventory and swapping battery
                    vehicle.pick_up(pick_up_bike)
                    
                # Perform all battery swaps
                for battery_swap_bike_id in action.battery_swaps:
                    battery_swap_bike = vehicle.cluster.get_bike_from_id(
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

                for helping_pickup_id in action.helping_pickup:
                    helping_pickup_bike = action.helping_cluster.get_bike_from_id(
                        helping_pickup_id
                    )

                    self.get_location_by_id(helping_pickup_bike.location_id).remove_bike(helping_pickup_bike)
                    
                    # Picking up bike and adding to vehicle inventory and swapping battery
                    vehicle.pick_up(helping_pickup_bike)

                if len(action.helping_delivery) > 0:
                    day = int((time // (60*24)) % 7)
                    hour = int((time // (60)) % 24)
                    for station in action.helping_cluster.areas:
                        while station.number_of_bikes() < station.get_target_state(day, hour) and len(vehicle.get_sb_bike_inventory()) > 0:
                            helping_delivery_id = action.helping_delivery.pop()

                            helping_delivery_bike = action.helping_cluster.get_bike_from_id(helping_delivery_id)
                            station.add_bike(helping_delivery_bike)
                            vehicle.drop_off(helping_delivery_id)

        # Moving the state/vehicle from this to next station
        vehicle.location = self.get_location_by_id(action.next_location)
        vehicle.cluster = action.cluster

        return refill_time

    def __repr__(self):
        string = f"<State: {len(self.get_parked_bikes())} bikes in {len(self.stations)} stations with {len(self.vehicles)} vehicles>\n"
        for station in self.get_locations():
            string += f"{str(station)}\n"
        for vehicle in self.get_locations():
            string += f"{str(vehicle)}\n"
        string += f"In use: {len(self.bikes_in_use.values())}"
        return string
    
    def get_closest_available_area(self, area, radius = FF_ROAMING_AREA_RADIUS):    
        available_area = None
        neighbors = area.neighbours

        while neighbors and not available_area:
            current_neighbor = neighbors.pop(0)  # Remove the first neighbor from the list to check it

            if len(current_neighbor.get_available_bikes()) > 0:
                available_area = current_neighbor
                break

            if radius > 0:
                # Extend with new neighbors not already in the list, avoiding duplicates
                for new_neighbor in current_neighbor.neighbours:
                    if new_neighbor not in neighbors and new_neighbor != area:
                        neighbors.append(new_neighbor)

            radius -= 1  # Decrease the radius after each neighbor check

            if not neighbors:  # If neighbors list is empty, break the loop
                break

        return available_area

    def get_neighbouring_stations(
        self,
        station: sim.Location,
        number_of_neighbours=None,
        is_sorted=True,
        exclude=None,
        not_full=False,
        not_empty=False
    ):
        """
        Get sorted list of stations closest to input station
        :param is_sorted: Boolean if the neighbours list should be sorted in a ascending order based on distance
        :param station: station to find neighbours for
        :param number_of_neighbours: number of neighbours to return
        :param exclude: neighbor ids to exclude
        :return:
        """
        neighbours = [
            state_station
            for state_station in self.get_stations()
            if state_station.location_id != station.location_id
            and state_station.location_id not in (exclude if exclude else [])
        ]
        if is_sorted:
            if not_full:
                neighbours = sorted(
                    [
                        state_station
                        for state_station in self.get_stations()
                        if state_station.location_id != station.location_id
                        and state_station.location_id not in (exclude if exclude else [])
                        and state_station.spare_capacity() >= 1
                    ],
                    key=lambda state_station: self.traveltime_matrix[(station.location_id, state_station.location_id)],
                )
            elif not_empty:
                neighbours = sorted(
                    [
                        state_station
                        for state_station in self.get_stations()
                        if state_station.location_id != station.location_id
                        and state_station.location_id not in (exclude if exclude else [])
                        and len(state_station.get_available_bikes()) >= 1
                    ],
                    key=lambda state_station: self.traveltime_matrix[(station.location_id, state_station.location_id)],
                )
            else:
                neighbours = sorted(
                    [
                        state_station
                        for state_station in self.get_stations()
                        if state_station.location_id != station.location_id
                        and state_station.location_id not in (exclude if exclude else [])
                    ],
                    key=lambda state_station: self.traveltime_matrix[(station.location_id, state_station.location_id)],
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
    
    def get_area_ids(self):
        return list(self.areas.keys())
    
    def get_areas(self):
        return list(self.areas.values())
    
    def get_station_ids(self):
        return list(self.stations.keys())
    
    def get_stations(self):
        return list(self.stations.values())
    
    def get_depot_ids(self):
        return list(self.depots.keys())
    
    def get_depots(self):
        return list(self.depots.values())
    
    def get_sb_locations(self):
        return {loc_id: loc for loc_id, loc in self.locations.items() if loc.is_station_based}
    
    def get_ff_locations(self):
        return {loc_id: loc for loc_id, loc in self.locations.items() if not loc.is_station_based}
    
    def get_closest_depot(self, vehicle):
        if vehicle.is_station_based is None:
            closest_depot = min(
                (depot for depot in self.get_depots()),
                key=lambda d: vehicle.location.distance_to(*d.get_location()),
                default=None
            )
        else:
            closest_depot = min(
                (depot for depot in self.get_depots() if depot.is_station_based == vehicle.is_station_based),
                key=lambda d: vehicle.location.distance_to(*d.get_location()),
                default=None
            )
        #TODO flytter seg ikke hvis det ikke finnes depot, men dette burde vel ikke skje?
        if not closest_depot:
            vehicle.battery_inventory = vehicle.battery_inventory_capacity
            return vehicle.location.location_id
        
        return closest_depot.location_id
    
    # TODO forstå denne
    def sample(self, sample_size: int):
        # Filter out bikes not in sample
        sampled_bike_ids = self.rng2.choice(
            [bike.bike_id for bike in self.get_parked_bikes()], sample_size, replace=False,
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
    
    def get_vehicles(self):
        return list(self.vehicles.values())
    
    def get_sb_vehicles(self):
        """
        Returns the vehicle object in the state corresponding to the vehicle id input
        :param vehicle_id: the id of the vehicle to fetch
        :return: vehicle object
        """
        return [vehicle for vehicle in self.get_vehicles() if vehicle.is_station_based]
    
    def get_ff_vehicles(self):
        """
        Returns the vehicle object in the state corresponding to the vehicle id input
        :param vehicle_id: the id of the vehicle to fetch
        :return: vehicle object
        """
        return [vehicle for vehicle in self.get_vehicles() if not vehicle.is_station_based]
    
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
