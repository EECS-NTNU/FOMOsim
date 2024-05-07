from policies.hlv_master.SB_BS_PILOT_policy import *
from policies.hlv_master.FF_BS_PILOT_policy import get_escooter_ids_load_swap
from settings import *
import sim
from .Visit import Visit
from .Plan import Plan
from .dynamic_clustering import Cluster, find_clusters, build_cluster
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
                congestion_criteria = CONGESTION_CRITERIA,
                starvation_criteria = STARVATION_CRITERIA,
                swap_threshold = BATTERY_LIMIT_TO_SWAP,
                operator_radius = OPERATOR_RADIUS
                 ):
        super().__init__(
            max_depth = max_depth,
            number_of_successors = number_of_successors, 
            time_horizon = time_horizon, 
            criticality_weights_set = criticality_weights_set, 
            evaluation_weights = evaluation_weights, 
            number_of_scenarios = number_of_scenarios, 
            discounting_factor = discounting_factor,
            congestion_criteria = congestion_criteria,
            starvation_criteria = starvation_criteria,
            swap_threshold = swap_threshold
        )
        self.operator_radius = operator_radius

    def get_best_action(self, simul, vehicle):
        """
        Returns an Action (with which bikes to swap batteries on, which bikes to pick-up, which bikes to unload, next location to drive to)

        Parameters:
        - simul = simulator
        - vehicle = Vehicle-object that is doing the action
        """
        action = super().get_best_action(simul, vehicle)

        start_help_time = time.time()
        pickup_escooter_ids = []
       
        # Find the areas of the stations visit
        current_area = simul.state.get_location_by_id(vehicle.location.area)
        next_area = simul.state.get_location_by_id(simul.state.get_location_by_id(action.next_location).area)

        current_cluster = Cluster(areas=[current_area], center_area=current_area)
        next_cluster = Cluster(areas=[next_area], center_area=next_area)
        current_cluster, _ = build_cluster([], current_cluster, self.operator_radius)
        next_cluster, _ = build_cluster([], next_cluster, self.operator_radius)

        current_deviation = current_cluster.get_max_num_usable(0, simul.time, simul.day(), simul.hour(), 0) - current_cluster.get_target_state(simul.day(), simul.hour())
        travel_time = simul.state.get_vehicle_travel_time(current_area.location_id, next_area.location_id)
        arrival_time = simul.time + travel_time
        day_arrival = int((arrival_time // (60*24)) % 7)
        hour_arrival = int((arrival_time // 60) % 24)
        next_deviation = next_cluster.get_max_num_usable(0, simul.time, simul.day(), simul.hour(), travel_time) - next_cluster.get_target_state(day_arrival, hour_arrival)

        if current_deviation > 0 and next_deviation < 0:
            num_escooters = min(current_deviation,
                                -next_deviation, 
                                vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()) - len(action.pick_ups)
                                )

            pickup_escooter_ids, _ = get_escooter_ids_load_swap(current_cluster, vehicle, num_escooters, 0, "pickup", 0)

        delivery_escooter_ids = [bike.bike_id for bike in vehicle.get_ff_bike_inventory()]
        
        simul.metrics.add_aggregate_metric(simul, "accumulated find helping action time", time.time() - start_help_time)
        simul.metrics.add_aggregate_metric(simul, "num helping escooter pickups", len(pickup_escooter_ids))
        simul.metrics.add_aggregate_metric(simul, "num helping escooter deliveries", len(delivery_escooter_ids))

        action.helping_pickup = pickup_escooter_ids
        action.helping_delivery = delivery_escooter_ids
        action.helping_cluster = current_cluster

        return action