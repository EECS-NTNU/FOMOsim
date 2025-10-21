from policies.hlv_master.collab3_X_PILOT_BS import Collab3
from policies.hlv_master.FF_BS_PILOT_policy import get_escooter_ids_load_swap
from policies.hlv_master.SB_BS_PILOT_policy import get_bike_ids_load_swap
from settings import *
import sim
from policies.hlv_master.Visit import Visit
from policies.hlv_master.Plan import Plan
from .dynamic_clustering import Cluster, build_cluster
import numpy as np
import time

class Collab4(Collab3): 
    def __init__(self, 
                max_depth = MAX_DEPTH, 
                number_of_successors = NUM_SUCCESSORS, 
                time_horizon = TIME_HORIZON, 
                criticality_weights_set = None,
                evaluation_weights = EVALUATION_WEIGHTS, 
                number_of_scenarios = NUM_SCENARIOS, 
                discounting_factor = DISCOUNTING_FACTOR,
                congestion_criteria = CONGESTION_CRITERIA,
                starvation_criteria = STARVATION_CRITERIA,
                swap_threshold = BATTERY_LIMIT_TO_SWAP,
                criticality_weights_set_ff = CRITICAILITY_WEIGHTS_SET_FF,
                criticality_weights_set_sb = CRITICAILITY_WEIGHTS_SET_SB,
                operator_radius = OPERATOR_RADIUS,
                num_clusters = MAX_NUMBER_OF_CLUSTERS,
                adjusting_criticality = ADJUSTING_CRITICALITY
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
            swap_threshold = swap_threshold,
            criticality_weights_set_ff= criticality_weights_set_ff,
            criticality_weights_set_sb= criticality_weights_set_sb,
            operator_radius = operator_radius,
            num_clusters = num_clusters,
            adjusting_criticality=adjusting_criticality
        )

    def get_best_action(self, simul, vehicle):
        action = super().get_best_action(simul, vehicle)
        

        current_location = vehicle.location if isinstance(vehicle.location, sim.Station) else vehicle.cluster
        next_location = simul.state.get_location_by_id(action.next_location)
        
        if isinstance(current_location, sim.Depot) or isinstance(next_location, sim.Depot):
            return action

        # Station -> Station
        if isinstance(current_location, sim.Station) and isinstance(next_location, sim.Station):
            start_help_time = time.time()
            helping_pickups = []

            # Find the areas of the stations visit
            current_area = simul.state.get_location_by_id(current_location.area)
            next_area = simul.state.get_location_by_id(next_location.area)

            current_cluster = Cluster([current_area], current_area, self.operator_radius, current_area.get_bikes(), current_area.get_neighbours())
            next_cluster = Cluster([next_area], next_area, self.operator_radius, next_area.get_bikes(), next_area.get_neighbours())
            current_cluster, _ = build_cluster([], current_cluster, self.operator_radius)
            next_cluster, _ = build_cluster([], next_cluster, self.operator_radius)

            current_deviation = current_cluster.get_max_num_usable(0, simul.time, simul.day(), simul.hour(), 0) - round(current_cluster.get_target_state(simul.day(), simul.hour()))
            travel_time = simul.state.get_vehicle_travel_time(current_area.location_id, next_area.location_id)
            arrival_time = simul.time + travel_time
            day_arrival = int((arrival_time // (60*24)) % 7)
            hour_arrival = int((arrival_time // 60) % 24)
            next_deviation = next_cluster.get_max_num_usable(0, simul.time, simul.day(), simul.hour(), travel_time) - round(next_cluster.get_target_state(day_arrival, hour_arrival))

            if current_deviation > 0 and next_deviation < 0:
                num_escooters = min(current_deviation, 
                                    -next_deviation, 
                                    vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()) - len(action.pick_ups)
                                    )
                helping_pickups, _ = get_escooter_ids_load_swap(current_cluster, vehicle, num_escooters, 0, "pickup", 0)
                
            helping_delivery = [bike.bike_id for bike in vehicle.get_ff_bike_inventory()]
            helping_cluster = current_cluster

            simul.metrics.add_aggregate_metric(simul, "accumulated find helping action time", time.time() - start_help_time)
            simul.metrics.add_aggregate_metric(simul, "num helping escooter pickups", len(helping_pickups))
            simul.metrics.add_aggregate_metric(simul, "num helping escooter deliveries", len(helping_delivery))
        
        # Station -> Area/Cluster
        elif isinstance(current_location, sim.Station) and not isinstance(next_location, sim.Station):
            start_help_time = time.time()
            current_area = simul.state.get_location_by_id(current_location.area)
            current_cluster = Cluster([current_area], current_area, self.operator_radius, current_area.get_bikes(), current_area.get_neighbours())
            current_cluster, _  = build_cluster([], current_cluster, self.operator_radius)

            next_deviation = len(next_location.get_available_bikes()) - round(next_location.get_target_state(simul.day(), simul.hour()))
            current_deviation = current_cluster.get_max_num_usable(0, simul.time, simul.day(), simul.hour(), 0) - round(current_cluster.get_target_state(simul.day(), simul.hour()))

            if current_deviation < 0:
                if next_deviation > 0:
                    helping_delivery = [bike.bike_id for bike in vehicle.get_ff_bike_inventory()]
                else:
                    excess_bikes = len(vehicle.get_ff_bike_inventory()) + next_deviation
                    if excess_bikes <= 0:
                        helping_delivery = []
                    else:
                        helping_delivery = [bike.bike_id for bike in vehicle.get_ff_bike_inventory()][:min(excess_bikes, current_deviation)]
            else:
                helping_delivery = []

            helping_pickups = []
            helping_cluster = current_cluster
            
            simul.metrics.add_aggregate_metric(simul, "accumulated find helping action time", time.time() - start_help_time)
            simul.metrics.add_aggregate_metric(simul, "num helping escooter deliveries", len(helping_delivery))
        
        # Area/Cluster -> Station
        elif not isinstance(current_location, sim.Station) and isinstance(next_location, sim.Station):
            start_help_time = time.time()
            current_stations = [simul.state.get_location_by_id(area.station) for area in vehicle.cluster.areas if area.station is not None] if vehicle.cluster is not None else []
            if current_stations != []:
                next_deviation = len(next_location.get_available_bikes()) - round(next_location.get_target_state(simul.day(), simul.hour()))
                current_deviations = [station.number_of_bikes() - round(station.get_target_state(simul.day(), simul.hour())) for station in current_stations]

                helping_delivery = []
                delivery_count = 0
                for deviation in current_deviations:
                    if deviation < 0:
                        if next_deviation > 0: # no need to take the next location action into consideration
                            excess_bikes = len(vehicle.get_sb_bike_inventory()) - delivery_count
                            if excess_bikes > 0:
                                delivery_count += min(excess_bikes, - deviation) 
                        else: # need to take the next location action into consideration
                            excess_bikes = len(vehicle.get_sb_bike_inventory()) + next_deviation - delivery_count
                            if excess_bikes > 0:
                                delivery_count += min(excess_bikes, - deviation)
                helping_delivery = [bike.bike_id for bike in vehicle.get_sb_bike_inventory()[:delivery_count]]
                
                station_bikes = {}
                for station in current_stations:
                    station_bikes.update(station.bikes)
                
                helping_cluster = Cluster(current_stations, vehicle.location, self.operator_radius, station_bikes, [])
                
                helping_pickups = []
            else:
                helping_pickups = []
                helping_delivery = []
                helping_cluster = None
            
            simul.metrics.add_aggregate_metric(simul, "accumulated find helping action time", time.time() - start_help_time)
            simul.metrics.add_aggregate_metric(simul, "num helping bike deliveries", len(helping_delivery))

        # Area/Cluster -> Area/Cluster
        else:
            # Helping policy added on 
            if action.cluster is None: # Ikke hjelp hvis du skal til depot
                return action
            
            start_help_time = time.time()
            current_stations = [simul.state.get_location_by_id(area.station) for area in vehicle.cluster.areas if area.station is not None] if vehicle.cluster is not None else []
            if current_stations != []:
                next_stations = [simul.state.get_location_by_id(area.station) for area in action.cluster.areas if area.station is not None]

                current_deviations = [station.number_of_bikes() - round(station.get_target_state(simul.day(), simul.hour())) for station in current_stations]
                next_deviations = [station.number_of_bikes() - round(station.get_target_state(simul.day(), simul.hour())) for station in next_stations]

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
                        helping_pickups += get_bike_ids_load_swap(current_stations[i], vehicle, round(current_stations[i].get_target_state(simul.day(), simul.hour())), num_pickup, "pickup", self.swap_threshold)[0] if (num_pickup > 0) else []
                        num_bikes -= num_pickup

                helping_delivery = [bike.bike_id for bike in vehicle.get_sb_bike_inventory()][:min(int(sum(current_deviations)), len(vehicle.get_sb_bike_inventory()))]
                
                helping_cluster = Cluster(current_stations, vehicle.location, self.operator_radius, {}, [])
            else:
                helping_pickups = []
                helping_delivery = []
                helping_cluster = None
            
            simul.metrics.add_aggregate_metric(simul, "accumulated find helping action time", time.time() - start_help_time)
            simul.metrics.add_aggregate_metric(simul, "num helping bike pickups", len(helping_pickups))
            simul.metrics.add_aggregate_metric(simul, "num helping bike deliveries", len(helping_delivery))

        # if len(set(helping_pickups)) != len(helping_pickups):
        #     print("Doble pickups")
        #     print(helping_pickups)
        # if len(set(helping_delivery)) != len(helping_delivery):
        #     print("Doble delivery")
        #     print(helping_delivery)
        
        # print(helping_pickups)
        # print(helping_delivery)
        # print(helping_cluster.get_bikes())

        action.helping_pickup =  helping_pickups
        action.helping_delivery = helping_delivery
        action.helping_cluster = helping_cluster

        return action