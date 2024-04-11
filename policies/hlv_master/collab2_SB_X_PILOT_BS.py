from policies.hlv_master.SB_BS_PILOT_policy import *
from policies.hlv_master.FF_BS_PILOT_policy import get_escooter_ids_load_swap
from settings import *
import sim
from .Visit import Visit
from .Plan import Plan
from .dynamic_clustering import Cluster, build_cluster_d, build_cluster_p
import time

class SB_Collab2(BS_PILOT):
    """
    Class for X-PILOT-BS policy. Finds the best action for e-bike systems.
    """
    def __init__(self, 
                max_depth = MAX_DEPTH, 
                number_of_successors = NUM_SUCCESSORS, 
                time_horizon = TIME_HORIZON, 
                criticality_weights_set = CRITICAILITY_WEIGHTS_SET, 
                evaluation_weights = EVALUATION_WEIGHTS, 
                number_of_scenarios = NUM_SCENARIOS, 
                discounting_factor = DISCOUNTING_FACTOR,
                overflow_criteria = OVERFLOW_CRITERIA,
                starvation_criteria = STARVATION_CRITERIA,
                swap_threshold = BATTERY_LIMIT_TO_SWAP
                 ):
        super().__init__(
            max_depth = max_depth,
            number_of_successors = number_of_successors, 
            time_horizon = time_horizon, 
            criticality_weights_set = criticality_weights_set, 
            evaluation_weights = evaluation_weights, 
            number_of_scenarios = number_of_scenarios, 
            discounting_factor = discounting_factor,
            overflow_criteria = overflow_criteria,
            starvation_criteria = starvation_criteria,
            swap_threshold = swap_threshold
        )

    def get_best_action(self, simul, vehicle):
        """
        Returns an Action (with which bikes to swap batteries on, which bikes to pick-up, which bikes to unload, next location to drive to)

        Parameters:
        - simul = simulator
        - vehicle = Vehicle-object that is doing the action
        """
        action = super().get_best_action(simul, vehicle)
        pickup_escooter_ids = []
       
        # Find the areas of the stations visit
        current_area = simul.state.get_location_by_id(vehicle.location.area)
        next_area = simul.state.get_location_by_id(simul.state.get_location_by_id(action.next_location).area)

        current_cluster = Cluster([current_area], current_area, current_area.get_bikes(), current_area.get_neighbours())
        next_cluster = Cluster([next_area], next_area, next_area.get_bikes(), next_area.get_neighbours())
        current_cluster, _ = build_cluster_p([], current_cluster, MAX_WALKING_AREAS, vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()) - len(action.pick_ups), simul)
        next_cluster, _ = build_cluster_d([], next_cluster, MAX_WALKING_AREAS, vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()) - len(action.pick_ups), simul)

        current_deviation = current_cluster.get_difference_from_target(simul.day(), simul.hour())
        next_deviation = next_cluster.get_difference_from_target(simul.day(), simul.hour())

        if current_deviation > 0 and next_deviation < 0:
            num_escooters = min(current_deviation, 
                                -next_deviation, 
                                vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()) - len(action.pick_ups)
                                )
            # print("Collab2, helping escooters =", num_escooters)
            pickup_escooter_ids, _ = get_escooter_ids_load_swap(current_cluster, vehicle, num_escooters, "pickup", 0) # since you can't swap
            # print(pickup_escooter_ids)
            # print(current_cluster, current_cluster.get_bikes())
            # print("original pickup action", action.pick_ups)

        delivery_escooter_ids = [bike.bike_id for bike in vehicle.get_ff_bike_inventory()]
        # if delivery_escooter_ids:
        #     print(delivery_escooter_ids)
        #     print(vehicle.bike_inventory)
        #     print("origianl delivery action", action.delivery_bikes)


        simul.metrics.add_aggregate_metric(simul, "Num helping pickups", len(pickup_escooter_ids))
        simul.metrics.add_aggregate_metric(simul, "Num helping deliveries", len(delivery_escooter_ids))

        action.helping_pickup = pickup_escooter_ids
        action.helping_delivery = delivery_escooter_ids
        action.helping_cluster = current_cluster

        return action

def make_cluster(radius, cluster):
    for _ in range(radius):
        new_neighbors = []
        for neighbor in cluster.get_neighbours():
            cluster.areas.append(neighbor)
            new_neighbors += [area for area in neighbor.get_neighbours() if area not in cluster.get_neighbours() and area not in new_neighbors and area not in cluster.areas]
        cluster.neighbours = new_neighbors
    return cluster