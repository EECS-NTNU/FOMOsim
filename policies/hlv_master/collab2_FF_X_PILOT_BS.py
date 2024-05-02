from policies.hlv_master.FF_BS_PILOT_policy import *
from policies.hlv_master.SB_BS_PILOT_policy import get_bike_ids_load_swap
from settings import *
import sim
from .Visit import Visit
from .Plan import Plan
from .dynamic_clustering import Cluster
import time

class FF_Collab2(BS_PILOT_FF): #Add default values from seperate setting sheme
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
        action = super().get_best_action(simul, vehicle)
        start_help_time = time.time()

        if action.cluster is None: # Ikke hjelp hvis du skal til depot
            return action

        # Helping policy added on 
        current_stations = [simul.state.get_location_by_id(area.station) for area in vehicle.cluster.areas if area.station is not None] if vehicle.cluster is not None else []
        next_stations = [simul.state.get_location_by_id(area.station) for area in action.cluster.areas if area.station is not None]

        current_deviations = [station.number_of_bikes() - station.get_target_state(simul.day(), simul.hour()) for station in current_stations]
        next_deviations = [station.number_of_bikes() - station.get_target_state(simul.day(), simul.hour()) for station in next_stations]

        helping_pickups = []
        if sum(current_deviations) > 0 and sum(next_deviations) < 0:
            # Number of SB-bikes to pickup
            num_bikes = min(
                sum(current_deviations),
                -sum(next_deviations),
                vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()) - len(action.pick_ups) # Left capacity in vehicle
            )
            for i in range(len(current_stations)):
                num_pickup = min(
                    max(0, current_deviations[i]),
                    num_bikes
                )
                helping_pickups += get_bike_ids_load_swap(current_stations[i], vehicle, num_pickup, "pickup")[0] if (num_pickup > 0) else []
                num_bikes -= num_pickup

        helping_delivery = [bike.bike_id for bike in vehicle.get_sb_bike_inventory()][:min(int(sum(current_deviations)), len(vehicle.get_sb_bike_inventory()))]
        
        helping_cluster = Cluster(current_stations, vehicle.location, {}, [])

        simul.metrics.add_aggregate_metric(simul, "accumulated find helping action time", time.time() - start_help_time)
        simul.metrics.add_aggregate_metric(simul, "num helping bike pickups", len(helping_pickups))
        simul.metrics.add_aggregate_metric(simul, "num helping bike deliveries", len(helping_delivery))

        action.helping_pickup = helping_pickups
        action.helping_delivery = helping_delivery
        action.helping_cluster = helping_cluster

        return action