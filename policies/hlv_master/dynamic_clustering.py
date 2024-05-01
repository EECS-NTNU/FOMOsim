import heapq
from sim.Location import Location
from shapely.geometry import MultiPoint
import numpy as np
import sim
from settings import *
import copy
import time
from collections import deque

def find_clusters(areas, n, max_length, vehicle, day, hour, operation):
    """
    Returns a list of n cluster-objects that are pickup clusters.

    Parameters:
    - areas = all area-objects to consider as centers
    - n = number of clusters to be considered
    - max_length = radius of areas
    - vehicle = vehicle-object that is doing the rebalancing
    - day, hour = time of the current event
    - operation = either delivery or pickup
    """
    tabu_list = []
    clusters = []
    all_possible_clusters = []

    # Makes a list of the n areas with the highest overflow of escooters
    if operation == "delivery":
        possible_areas = heapq.nlargest(n, areas, key=lambda area: -area.get_difference_from_target(day, hour, CLUSTER_USE_NEIGHBOURS))
    elif operation == "pickup":
        possible_areas = heapq.nlargest(n, areas, key=lambda area: area.get_difference_from_target(day, hour, CLUSTER_USE_NEIGHBOURS))
    elif operation == "both":
        possible_areas = heapq.nlargest(n//2, areas, key=lambda area: -area.get_difference_from_target(day, hour, CLUSTER_USE_NEIGHBOURS))
        possible_areas += heapq.nlargest(n//2, areas, key=lambda area: area.get_difference_from_target(day, hour, CLUSTER_USE_NEIGHBOURS))

    # Make clusters for all, with the area as a center, if the area are not already added in another cluster
    for area in possible_areas:
        if area not in tabu_list:
            c = Cluster(areas=[area], center_area=area, bikes=area.get_bikes(), neighbours=area.get_neighbours())
            
            # Expands the cluster, looking at areas with needing to have escooters picked up
            c, tabu_list = build_cluster(tabu_list, c, max_length)

            # Only add the cluster into potenial clusters if it doesn't balance it self out by it's neighbors, but have too many bikes
            difference = c.get_max_num_usable(vehicle) - c.get_target_state(day, hour)

            if (difference > 0 and operation == "pickup") or (difference < 0 and operation == "delivery") or (difference != 0 and operation == "both"):
                clusters.append(c)
            else:
                all_possible_clusters.append(c)

    return clusters

def build_cluster(tabu_list, c, max_depth):
    # Condition to limit the radius of the cluster
    neighbours = deque(c.center_area.get_neighbours())
    for _ in range(max_depth):
        new_neighbours = []
        while neighbours:
            n = neighbours.popleft()
            if n not in c.areas:
                c.add_area(n)
                new_neighbours.extend(new_n for new_n in n.get_neighbours() if new_n not in neighbours and new_n not in c.areas)
        neighbours = deque(new_neighbours)
    
    if len(c.areas) > 127:
        print("for mange neighbours", c.location_id, len(c.areas))

    return c, tabu_list

class Cluster(Location):
    def __init__(
        self,
        areas,
        center_area,
        bikes = [], #dict, key = bike_id, value = object
        neighbours = []
    ):
        super().__init__(
            *(center_area.get_location() if center_area.get_location() else self.__compute_center(center_area.border_vertices)), center_area.location_id
        )

        self.center_area = center_area
        self.areas = areas
        self.neighbours = neighbours
        self.not_included = []

    def add_area(self, area):
        if area not in self.areas:
            self.areas.append(area)

    def add_not_included_neighbor(self, area):
        if area not in self.not_included:
            self.not_included.append(area)

    def get_bike_ids(self):
        return [bike.bike_id for area in self.areas for bike in area.bikes.values()]
    
    def get_bikes(self):
        """
        Used to find which IDs to pick up, so this has to only include the bikes we want to pick up
        """
        return [bike for area in self.areas for bike in area.get_bikes()]

    def get_max_num_usable(self, vehicle):
        return len(self.get_available_bikes()) + min(len(self.get_unusable_bikes()), vehicle.battery_inventory)
    
    def get_neighbours(self):
        neighbours = {neighbor.location_id: neighbor for area in self.areas 
                                                     for neighbor in area.get_neighbours() 
                                                     if neighbor not in self.areas}
        return list(neighbours.values())
    
    def number_of_bikes(self):
        return len(self.get_bikes())
    
    def get_swappable_bikes(self, battery_limit=BATTERY_LIMIT_TO_SWAP):
        """
        Filter out bikes with 100% battery and sort them by battery percentage
        """
        bikes = [
            bike for bike in self.get_bikes() if bike.hasBattery() and bike.battery < battery_limit
        ]
        return sorted(bikes, key=lambda bike: bike.battery, reverse=False)

    def get_bike_from_id(self, bike_id):
        #TODO lag dict og hent
        if not isinstance(bike_id, str):
            print("Not an ID:", bike_id)
            print(bike_id.bike_id)

        for bike in self.get_bikes():
            if bike.bike_id == bike_id:
                return bike
        print(bike_id, [bike.bike_id for bike in self.get_bikes()])
        return None
    
    def get_difference_from_target(self, day, hour):
        """
        TODO
        skal denne brukes i hjelpe-policy?

        Returns the difference from target state for all areas in the cluster 
        and the areas not included in the cluster, as they are in allowed walking distance and can absorb demand.

        If returning positive number -> too many bikes and needs to have some picked up
        If returning negative number -> too few bikes, needs to have some delivered
        """
        all_areas = self.areas
        target_state = sum(area.get_target_state(day, hour) for area in all_areas)
        cluster_inventory = sum(len(area.get_available_bikes()) for area in all_areas)
        # print(f'Target state: {target_state}, Available bikes: {cluster_inventory}')
        return cluster_inventory - target_state
    
    def get_target_state(self, day, hour):
        target_state = sum(a.get_target_state(day,hour) for a in (self.areas))
        return target_state

    def get_arrive_intensity(self, day, hour):
        arrive_intensity = sum(a.get_arrive_intensity(day, hour) for a in self.areas + self.not_included)
        return arrive_intensity

    def get_leave_intensity(self, day, hour):
        leave_intensity = sum(a.get_leave_intensity(day,hour) for a in self.areas + self.not_included)
        return leave_intensity

    def get_available_bikes(self):
        return [
            bike for bike in self.get_bikes() if bike.usable()
        ]
    
    def get_unusable_bikes(self):
        return [
            bike for bike in self.get_bikes() if not bike.usable()
        ]

    def get_swappable_bikes(self, battery_limit = BATTERY_LIMIT_TO_SWAP):
        bikes = [
            bike for bike in self.get_bikes() if bike.hasBattery() and bike.battery < battery_limit
        ]
        return sorted(bikes, key=lambda bike: bike.battery, reverse=False)
    
    def __repr__(self):
        return (
            f'Cluster: (Center = {self.center_area.location_id}, Areas = {len(self.areas)} (not_included = {len(self.not_included)}), Bikes = {len(self.get_bikes())} (avail = {len(self.get_available_bikes())})'
        )