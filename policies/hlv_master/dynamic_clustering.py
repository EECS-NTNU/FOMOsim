import heapq
from sim.Location import Location
from shapely.geometry import MultiPoint
import numpy as np
import sim
from settings import *
import copy

def clusterPickup(areas, n, threshold, max_radius, vehicle, simul):
    """
    Returns list of n Cluster-objects based on the n-areas with the most escooters

    Parameters:
    - areas = all areas evaluated to make clusters
    - n = number of clusters to make
    - threshold = number of bikes in the area has to be higher to be included in a cluster
    - max_length = number of clusters from center allowed to explore within one cluster, excluding the center (1 = center + it's neighbours)
    - vehicle = Vehicle-object that is making the operations
    - simul = Simulator
    """
    highest_density_areas = heapq.nlargest(n, areas, key=lambda area: area.number_of_bikes())
    tabu_list = []
    clusters = []

    remaining_vehicle_capacity = vehicle.bike_inventory_capacity - len(vehicle.bike_inventory)

    # Makes each area the center of a cluster, and builds it if not in the tabu-list
    for area in highest_density_areas:
        if area not in tabu_list: # TODO oppdaterer denne tabu_listen seg?
            cluster = Cluster([area], area, area.bikes, area.get_neighbours())
            build_cluster_p(tabu_list, cluster, max_radius, remaining_vehicle_capacity, threshold, 1, simul)
            clusters.append(cluster)

    return clusters


def build_cluster_p(tabu_list, cluster, max_radius, remaining_vehicle_capacity, threshold, counter, simul):
    """
    
    """
    # Loop until the counter exceeds the maximum depth allowed for clustering
    while counter <= max_radius:
        new_neighbours = []

        # Iterate over each neighbor of a current cluster
        for neighbour in cluster.get_neighbours():

            # Stop adding to cluster if the amount of bikes to pick up exceeds the vehicle capacity
            cluster_overflow = len(cluster.bikes) - cluster.get_target_state(simul.day(), simul.hour())
            if cluster_overflow >= remaining_vehicle_capacity:
                break

            # If the neighbouring area does not have enough bikes, the area is considered the clusters neighbor, but is not added in the cluster
            if neighbour.number_of_bikes() < threshold: # TODO burde ikke dette være i forhold til target state??
                new_neighbours.append(neighbour)
                continue

            # Add area to cluster and tabu_list to not be checked further
            cluster.areas.append(neighbour)
            tabu_list.append(neighbour)

            # Add all bikes from this neighbor to the cluster's bike list
            cluster.bikes.update({bike.bike_id: bike for bike in neighbour.bikes})

            # Extend new_neighbours with neighbors of the current neighbor that haven't been considered yet and aren't in tabu list
            new_neighbours.extend([area for area in neighbour.get_neighbours() if area not in cluster.get_neighbours() and area not in tabu_list and area not in new_neighbours])

        counter += 1

        cluster.neighbors = new_neighbours  # Update the cluster's neighbors list with the newly found neighbors

        # Recursively call the function to build cluster parts, presumably to continue expanding or
        build_cluster_p(tabu_list, cluster, max_radius, counter, remaining_vehicle_capacity, threshold, simul)



def clusterDelivery(areas, n, threshold, max_lenght, veichle, simul):
    tabu_list = []
    clusters = []

    largest_shortfall_areas = heapq.nlargest(n, areas, key=lambda area: area.number_of_bikes() - area.get_target_state(simul.day(), simul.hour()))

    cut_off = len(veichle.bike_inventory)

    for area in largest_shortfall_areas:
        if area not in tabu_list:
            c = Cluster([area], area, area.bikes, area.get_neighbours())

            build_cluster_d(tabu_list, c, max_lenght, cut_off, threshold, 1, simul)    

            
            clusters.append(c)
    
    return clusters


def build_cluster_d(tabu_list, c, max_depth, cut_off, threshold, counter, simul):
    while counter <= max_depth:
        new_neighbours = []
        for neighbour in c.get_neighbours():
                if (c.get_target_state(simul.day(), simul.hour())) - len(c.bikes) >= cut_off:
                    break
                if (neighbour.number_of_bikes() - neighbour.get_target_state(simul.day(), simul.hour())) < threshold:
                    continue
                c.areas.append(neighbour)
                tabu_list.append(neighbour)
                for bike_id, bike in neighbour.bikes.items():
                    c.bikes[bike_id] = bike
                new_neighbours += [area for area in neighbour.get_neighbours() if area not in c.get_neighbours() and area not in tabu_list and area not in new_neighbours]

        
        counter += 1
        c.neighbors = new_neighbours
        build_cluster_d(tabu_list, c, max_depth, counter, cut_off, threshold, simul)

class Cluster(Location):
    """
    Class for a cluster of areas that is being operated at the same time.

    Parameters:
    - areas = list of Area-objects
    - center_area = the center area-object of which the cluster is based on
    - bikes = dictionary of all bikes within the cluster
    - neighbors = list of area-objects that border the cluster
    """
    def __init__(
        self,
        areas,
        center_area,
        bikes = {}, #dict, key = bike_id, value = escooter-object
        neighbours = []
    ):
        super().__init__(
            *(center_area.get_location() if center_area.get_location() else self.__compute_center(center_area.border_vertices)), center_area.location_id
        )

        self.center_area = center_area
        self.areas = areas
        self.neighbours = neighbours
        self.bikes = bikes
        
    def __compute_center(self, border_vertices):
        sum_lon = sum(v[0] for v in border_vertices)
        sum_lat = sum(v[1] for v in border_vertices)

        n = len(border_vertices)
        centroid_lat = sum_lat/n
        centroid_lon = sum_lon/n

        return centroid_lat, centroid_lon

    def set_bikes(self, bikes):
        self.bikes = {bike.bike_id : bike for bike in bikes}

    def get_neighbours(self):
        return self.neighbours
    
    def number_of_bikes(self):
        return len(self.bikes)
    
    def get_swappable_bikes(self, battery_limit=BATTERY_LIMIT_TO_SWAP):
        """
        Returns a list of bikes with battery under battery limit and sort them by ascending battery percentage

        Parameters:
        - battery_limit = the battery limit the bikes returned have to be under
        """
        bikes = [
            bike for bike in self.bikes.values() if bike.hasBattery() and bike.battery < battery_limit
        ]
        return sorted(bikes, key=lambda bike: bike.battery, reverse=False)

    def get_bike_from_id(self, bike_id):
        return self.bikes[bike_id]
    
    def get_target_state(self, day, hour):
        target_state = sum(a.get_target_state(day,hour) for a in self.areas)
        return target_state

    def get_arrive_intensity(self, day, hour):
        arrive_intensity = sum(a.get_arrive_intensity(day, hour) for a in self.areas)
        return arrive_intensity

    def get_leave_intensity(self, day, hour):
        leave_intensity = sum(a.get_leave_intensity(day,hour) for a in self.areas)
        return leave_intensity

    def get_available_bikes(self):
        """
        Returns list of bike-objects with sufficient battery level for rental
        """
        return [
            bike for bike in self.bikes.values() if bike.usable()
        ]
    
    def __repr__(self):
        return (
            f"<Cluster: {len(self.bikes)} bikes>, {len(self.areas)} areas"
        )

    def __str__(self):
        return f"Cluster {len(self.areas)}: Ideal {self.get_target_state(0, 8):3d} Bikes {len(self.bikes):3d}"
