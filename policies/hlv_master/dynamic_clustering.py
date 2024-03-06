import heapq
from sim.Location import Location
from shapely.geometry import MultiPoint
import numpy as np
import sim
from settings import *
import copy

#areas = list of all areas
#n = maximum number of clusters
#threshold = threshold for adding neighboughring areas to cluster
#veichle = current veichle 
#need to add max_lenght

#Now this only works for neighbours - need to add exploration / layers - rekrkusjon?

def clusterPickup(areas, n, threshold, max_lenght, vehicle, simul):
    #areas sorted based on density 
    highest_density_areas = heapq.nlargest(n, areas, key=lambda area: area.number_of_bikes())
    #tabu-list
    tabu_list = []
    clusters = []

    cut_off = vehicle.bike_inventory_capacity - len(vehicle.bike_inventory)

    for area in highest_density_areas:
        if area not in tabu_list:
            c = Cluster([area],area, area.bikes, area.get_neighbours())

            build_cluster_p(tabu_list, c, max_lenght, cut_off, threshold, 1, simul)    

            
            clusters.append(c)
            
    
    return clusters


def build_cluster_p(tabu_list, c, max_depth, cut_off, threshold, counter, simul):
    while counter <= max_depth:
        new_neighbours = []
        for neighbour in c.get_neighbours():
                if (len(c.bikes) - c.get_target_state(simul.day(), simul.hour()))>=cut_off:
                    break
                #if (neighbour in tabu_list):
                #    continue
                if neighbour.number_of_bikes() < threshold:
                    new_neighbours.append(neighbour)
                    continue
                c.areas.append(neighbour)
                tabu_list.append(neighbour)
                # [[sum(x) for x in zip(*t)] for t in zip(c.leave_intensities, neighbour.leave_intensity)] 
                # c.leave_intensities += neighbour.get_leave_intensity()
                # c.arrive_intensities += neighbour.get_arrive_intensity()
                # c.target_state += neighbour.get_target_state()
                for bike in neighbour.bikes:
                    c.bikes[bike.bike_id] = bike
                new_neighbours += [area for area in neighbour.get_neighbours() if area not in c.get_neighbours and area not in tabu_list and area not in new_neighbours]
                #c.neighbours.remove(neighbour)

        
        counter += 1

        c.neighbors = new_neighbours

        build_cluster_p(tabu_list, c, max_depth, counter, cut_off, threshold, simul)


def clusterDelivery(areas, n, threshold, max_lenght, veichle, simul):
    tabu_list = []
    clusters = []

    largest_shortfall_areas = heapq.nlargest(n, areas, key=lambda area: area.number_of_bikes() - area.get_target_state(simul.day(), simul.hour()))

    cut_off = len(veichle.bike_inventory)

    for area in largest_shortfall_areas:
        if area not in tabu_list:
            c = Cluster([area],area, area.bikes, area.get_neighbours())

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
                #c.neighbours.remove(neighbour)

        
        counter += 1
        c.neighbors = new_neighbours
        build_cluster_d(tabu_list, c, max_depth, counter, cut_off, threshold, simul)




        



            







#Cluster object:
#Areas - List of areas within cluster
#Center_area -  one area 
#Leave_intensity
#Arrive_intensity
#Density
#Target state

class Cluster(Location):
    def __init__(
        self,
        areas,
        center_area,
        bikes = {}, #dict, key = bike_id, value = object
        neighbours = []
    ):
        super().__init__(
            *(center_area.get_location() if center_area.get_location() else self.__compute_center(center_area.border_vertices)), center_area.location_id
        )

        self.center_area = center_area
        self.areas = areas
        self.neighbours = neighbours
        self.bikes = bikes
        

    def set_bikes(self, bikes):
        self.bikes = {bike.bike_id : bike for bike in bikes}

    def get_neighbours(self):
        return self.neighbours
    
    def number_of_bikes(self):
        return len(self.bikes)
    
    def get_swappable_bikes(self, battery_limit=BATTERY_LIMIT_TO_SWAP):
        """
        Filter out bikes with 100% battery and sort them by battery percentage
        """
        bikes = [
            bike for bike in self.bikes.values() if bike.hasBattery() and bike.battery < battery_limit
        ]
        return sorted(bikes, key=lambda bike: bike.battery, reverse=False)

    def get_bike_from_id(self, bike_id):
        return self.bikes[bike_id]
    
    def __compute_center(self, border_vertices):
        sum_lon = sum(v[0] for v in border_vertices)
        sum_lat = sum(v[1] for v in border_vertices)

        n = len(border_vertices)
        centroid_lat = sum_lat/n
        centroid_lon = sum_lon/n

        return centroid_lat, centroid_lon
    
    def get_target_state(self, day, hour):
        target_state = 0
        for a in self.areas:
            target_state += a.get_target_state(day,hour)
        return target_state

    def get_arrive_intensity(self, day, hour):
        arrive_intensity = 0
        for a in self.areas:
            arrive_intensity += a.get_arrive_intensity(day, hour)
        return arrive_intensity

    def get_leave_intensity(self, day, hour):
        leave_intensity = 0
        for a in self.areas:
            leave_intensity += a.get_leave_intensity(day,hour)
        return leave_intensity

    def get_available_bikes(self):
        return [
            bike for bike in self.bikes.values() if bike.usable()
        ]

    def get_swappable_bikes(self, battery_limit = BATTERY_LIMIT_TO_SWAP):
        bikes = [
            bike for bike in self.bikes.values() if bike.hasBattery() and bike.battery < battery_limit
        ]
        return sorted(bikes, key=lambda bike: bike.battery, reverse=False)