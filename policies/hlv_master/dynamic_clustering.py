import heapq
from sim.Location import Location
from shapely.geometry import MultiPoint
import numpy as np
import sim
from settings import *
import copy
import time

def clusterPickup(areas, n, max_lenght, vehicle, simul):
    """
    Returns a list of n cluster-objects that are pickup clusters.

    Parameters:
    - areas = all area-objects to consider as centers
    - n = number of clusters to be considered
    - max_lenght = radius of areas
    - vehicle = vehicle-object that is doing the rebalancing
    """
    # Makes a list of the n areas with the highest overflow of escooters
    highest_density_areas = heapq.nlargest(n, areas, key=lambda area: area.get_difference_from_target(simul.day(), simul.hour(), CLUSTER_USE_NEIGHBOURS))
    
    tabu_list = []
    clusters = []
    all_possible_clusters = []
    cut_off = vehicle.bike_inventory_capacity - len(vehicle.bike_inventory)
    
    # Make clusters for all, with the area as a center, if the area are not already added in another cluster
    for area in highest_density_areas:
        if area not in tabu_list:
            c = Cluster(
                areas=[area], 
                center_area=area, 
                bikes=area.get_bikes(), 
                neighbours=area.get_neighbours())
            
            # Expands the cluster, looking at areas with needing to have escooters picked up
            c, tabu_list = build_cluster_p(tabu_list, c, max_lenght, cut_off, simul)

            # Only add the cluster into potenial clusters if it doesn't balance it self out by it's neighbors, but have too many bikes
            difference = c.get_max_num_usable(vehicle) - c.get_target_state(simul.day(), simul.hour())
            if difference > 0:
                clusters.append(c)
            else:
                all_possible_clusters.append(c)

    return clusters

def build_cluster_p(tabu_list, c, max_depth, cut_off, simul):
    # Condition to limit the radius of the cluster
    for i in range(max_depth):
        neighbours = c.get_neighbours()
        for neighbour in neighbours:
            # if neighbor doesn't have an overload of escooters, do not add to cluster
            if len(neighbour.get_bikes()) - neighbour.get_target_state(simul.day(), simul.hour()) <= 0:
                c.add_not_included_neighbor(neighbour)

            c.add_area(neighbour)
            tabu_list.append(neighbour)

    return c, tabu_list

def clusterDelivery(areas, n, max_lenght, vehicle, simul):
    """
    Returns a list of n cluster-objects that are pickup clusters.

    Parameters:
    - areas = all area-objects to consider as centers
    - n = number of clusters to be considered
    - max_lenght = radius of areas
    - vehicle = vehicle-object that is doing the rebalancing
    """

    # Initialize
    tabu_list = []
    clusters = []
    all_possible_clusters = []
    cut_off = len(vehicle.bike_inventory)

    start_time = time.time()
    # Make list of n areas that are lacking the most
    largest_shortfall_areas = heapq.nlargest(n, areas, key=lambda area: len(area.get_available_bikes()) - area.get_target_state(simul.day(), simul.hour()))
    sorted_time = time.time()

    # Make clusters for all, with the area as a center, if the area are not already added in another cluster
    for area in largest_shortfall_areas:
        if area not in tabu_list:
            c = Cluster([area], area, area.get_bikes(), area.get_neighbours())
            c, tabu_list = build_cluster_d(tabu_list, c, max_lenght, cut_off, simul)
            
            difference = c.get_max_num_usable(vehicle) - c.get_target_state(simul.day(), simul.hour())
            if difference < 0:
                clusters.append(c)
            else:
                all_possible_clusters.append(c)
    
    # print(f'Duration of sorting: {sorted_time - start_time}, Duration of making d clusters: {time.time() - sorted_time}')
    
    # if all(len(cluster.areas) == 1 for cluster in clusters):
        # print("ingen d clustere som expander", clusters)
        # print("all possible d c:", all_possible_clusters)

    return clusters

# TODO burde ikke alt bare legges til, også sjekke om det finnes områder som har for lite i total, ikke bare enkelte areas?
def build_cluster_d(tabu_list, c, max_depth, cut_off, simul):
    for i in range(max_depth):
        neighbours = c.get_neighbours()
        for neighbour in neighbours:
            # Only add as neighbor, not in cluster, if area does not need more bikes
            if (len(neighbour.get_bikes()) - neighbour.get_target_state(simul.day(), simul.hour())) >= 0:
                c.add_not_included_neighbor(neighbour)

            c.add_area(neighbour)
            tabu_list.append(neighbour)
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