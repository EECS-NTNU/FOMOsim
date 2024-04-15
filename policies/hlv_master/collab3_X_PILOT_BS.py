from policies.hlv_master.FF_BS_PILOT_policy import BS_PILOT_FF, get_max_num_usable_escooters, get_escooter_ids_load_swap as get_escooter_ids_load_swap_ff
from policies.hlv_master.SB_BS_PILOT_policy import get_bike_ids_load_swap
from settings import *
import sim
from policies.hlv_master.Visit import Visit
from policies.hlv_master.Plan import Plan
from .collab3_Criticality_Score import calculate_criticality_ff, calculate_criticality_sb
from policies.hlv_master.Simple_calculations import copy_arr_iter
from policies.hlv_master.Simple_calculations import calculate_net_demand
from policies.hlv_master.dynamic_clustering import clusterDelivery, clusterPickup
import numpy as np
import time

class Collab3(BS_PILOT_FF):
    def __init__(self, 
                max_depth = MAX_DEPTH, 
                number_of_successors = NUM_SUCCESSORS, 
                time_horizon = TIME_HORIZON, 
                criticality_weights_set = None,
                evaluation_weights = EVALUATION_WEIGHTS, 
                number_of_scenarios = NUM_SCENARIOS, 
                discounting_factor = DISCOUNTING_FACTOR,
                overflow_criteria = OVERFLOW_CRITERIA,
                starvation_criteria = STARVATION_CRITERIA,
                swap_threshold = BATTERY_LIMIT_TO_SWAP,
                criticality_weights_set_ff = CRITICAILITY_WEIGHTS_SET,
                criticality_weights_set_sb = CRITICAILITY_WEIGHTS_SET
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
        self.criticality_weights_set_ff = criticality_weights_set_ff
        self.criticality_weights_set_sb = criticality_weights_set_sb
    
    def get_best_action(self, simul, vehicle):
        """
        Returns an Action (with which escooters to swap batteries on, which bikes to pick-up, which bikes to unload, next location to drive to)
        Additionally, it returns the cluster for which the next location is a part of.

        Parameters:
        - simul = simulator
        - vehicle = Vehicle-object that is doing the action
        """
        start_logging_time = time.time() 
        next_location = None
        bikes_to_pickup = []
        bikes_to_deliver = []
        batteries_to_swap = []

        end_time = simul.time + self.time_horizon 
        total_num_bikes_in_system = len(simul.state.get_all_bikes())

        # Goes to depot if the vehicle's battery inventory is empty on arrival, and picks up all escooters at location that is unusable
        if vehicle.battery_inventory <= 0 and len(simul.state.depots) > 0:
            next_location = simul.state.get_closest_depot(vehicle)
            # If no depot, just stay and do nothing
            if next_location == vehicle.location.location_id:
                return sim.Action(
                [],
                [],
                [],
                next_location
            )
            bikes_to_pickup = vehicle.location.get_swappable_bikes()
            max_pickup = min(vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory()), len(bikes_to_pickup))
            return sim.Action(
                [],
                bikes_to_pickup[:max_pickup],
                [],
                next_location
            )

        # Loading and swap strategy at current area or cluster is always chosen greedily
        if isinstance(vehicle.location, sim.Depot): # No action needed if at depot
            bikes_to_pickup = []
            bikes_to_deliver = []
            batteries_to_swap = []
        elif vehicle.cluster is None:
            # TODO sende inn congestion_criteria?
            bikes_to_pickup, bikes_to_deliver, batteries_to_swap = calculate_loading_quantities_and_swaps_greedy(vehicle, simul, vehicle.location, self.overflow_criteria, self.starvation_criteria, self.swap_threshold)
            num_bikes_pickup = len(bikes_to_pickup)
            num_bikes_deliver = len(bikes_to_deliver)
            num_batteries_to_swap = len(batteries_to_swap)
        else:
            bikes_to_pickup, bikes_to_deliver, batteries_to_swap = calculate_loading_quantities_and_swaps_greedy(vehicle, simul, vehicle.cluster, self.overflow_criteria, self.starvation_criteria, self.swap_threshold)
            num_bikes_pickup = len(bikes_to_pickup)
            num_bikes_deliver = len(bikes_to_deliver)
            num_batteries_to_swap = len(batteries_to_swap)

        # Make a plan for all vehicles
        # Dictionary with key = vehicle_id, 
        #   value = List of Visits, where the first one is the current location of vehicle, or the location the vehicle is on its way to
        plan_dict = dict()
        for v in simul.state.get_vehicles():
            # If vehicle is at a location, add current location to the plan with the greedy loading and swap strategy
            if v.eta == 0:
                plan_dict[v.vehicle_id] = [Visit(v.location, num_bikes_pickup, num_bikes_deliver, num_batteries_to_swap, simul.time, v)]
            
            # If the vehicle is driving, use pilot to calculate the loading and swap strategy and add to the plan
            else:
                num_bikes_pickup, num_bikes_deliver, num_batteries_to_swap = self.calculate_loading_quantities_and_swaps_pilot(v, simul, v.location, v.eta) #I think eta is estimated time of arrival
                plan_dict[v.vehicle_id] = [Visit(v.location, int(num_bikes_pickup), int(num_bikes_deliver), int(num_batteries_to_swap), v.eta, v)]
        
        # All locations the vehicles are at or are on their way to is added to the tabu list and plan
        tabu_list = [v.location.location_id for v in simul.state.get_vehicles()]
        plan = Plan(plan_dict, tabu_list)

        # Use X-PILOT-BS to find which location to drive to next
        next_location, cluster = self.PILOT_function(simul, 
                                                     vehicle,
                                                     plan, 
                                                     self.max_depth, 
                                                     self.number_of_successors, 
                                                     end_time, 
                                                     total_num_bikes_in_system)
        
        simul.metrics.add_aggregate_metric(simul, "Accumulated solution time", time.time()-start_logging_time)
        simul.metrics.add_aggregate_metric(simul, 'Number of get_best_action', 1)

        return sim.Action(
            batteries_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_location,
            cluster
        )

    def PILOT_function(self, simul, vehicle, initial_plan, max_depth, number_of_successors, end_time, total_num_bikes_in_system):
        completed_plans = []
        for i in range(len(self.criticality_weights_set_ff)):
            num_successors = number_of_successors
            plans = [[] for i in range(max_depth +1)] # [[], [], [], ... ] -> len() = max_depth+1
            plans[0].append(initial_plan) # plan of visits so far
            depths = [i for i in range (1, max_depth+1)] # [1, 2, 3, ... ] -> len() = max_depth
            weight_set = [self.criticality_weights_set_ff[i], self.criticality_weights_set_sb[i]]

            for depth in depths:
                if depth == 2 or depth == 3:
                    num_successors = max(1, round(num_successors/2)) # halve beta after depth 1 and 2
                
                while plans[depth-1] != []:
                    plan = plans[depth-1].pop(0)
                    next_vehicle = plan.next_visit.vehicle
                    if next_vehicle != vehicle:
                        num_successors_other_veheicle = max(1, round(num_successors/2))
                        new_visits = self.greedy_next_visit(plan, simul, num_successors_other_veheicle, weight_set, total_num_bikes_in_system)
                    else:
                        new_visits = self.greedy_next_visit(plan, simul, num_successors, weight_set, total_num_bikes_in_system)
                    
                    if new_visits == None or plan.next_visit.get_depature_time() > end_time:
                        new_plan = Plan(plan.copy_plan(),copy_arr_iter(plan.tabu_list), weight_set, plan.branch_number)
                        plans[depth].append(new_plan)
                    else:
                        for branch_number, visit in enumerate(new_visits):
                            new_plan_dict = plan.copy_plan()
                            new_plan_dict[next_vehicle.vehicle_id].append(visit)
                            tabu_list = copy_arr_iter(plan.tabu_list)
                            tabu_list.append(visit.station.location_id)

                            if depth == 1:
                                new_plan = Plan(new_plan_dict, tabu_list, weight_set, branch_number)
                            else:
                                new_plan = Plan(new_plan_dict, tabu_list, weight_set, plan.branch_number)
                            

                            if next_vehicle.vehicle_id == vehicle.vehicle_id:
                                plans[depth].append(new_plan)
                            else:
                                plans[depth-1].append(new_plan)


            for plan in plans[max_depth]:

                dep_time = plan.next_visit.get_depature_time()
                temp_plan = Plan(plan.copy_plan(), copy_arr_iter(plan.tabu_list), weight_set, plan.branch_number)

                while dep_time < end_time:
                    new_visit = self.greedy_next_visit(temp_plan, simul, 1, weight_set, total_num_bikes_in_system)
                    
                    if new_visit != None:
                        new_visit = new_visit[0]
                        temp_plan.tabu_list.append(new_visit.station.location_id)
                    else:
                        break

                    
                    temp_plan.plan[temp_plan.next_visit.vehicle.vehicle_id].append(new_visit)
                    dep_time = new_visit.get_depature_time()
                    temp_plan.find_next_visit()
                
                completed_plans.append(temp_plan)
            
        plan_scores = dict()

        #Generates different demand scenarios?
        scenarios = self.generate_scenarioes(simul, self.number_of_scenarios, poisson = True)

        for plan in completed_plans:
            plan_scores[plan] = []
            for scenario_dict in scenarios:
                score = 0
                for v in plan.plan:
                    score += self.evaluate_route(plan.plan[v], scenario_dict, end_time, simul, self.evaluation_weights, total_num_bikes_in_system)
                plan_scores[plan].append(score)
        
        # Returns the station with the best avarage score over all scenarios
        return self.return_best_move_avarage(vehicle, simul, plan_scores)

    def calculate_loading_quantities_and_swaps_pilot(self, vehicle, simul, location, eta):
        """
        Returns the NUMBER of bikes to pick up, deliver and swap batteries on. Takes future demand into account by calculating it for the next hour, and treats it as evenly distributed throughout that hour

        Parameters:
        - vehicle = Vehicle-object the action is considered for
        - simul = Simulator
        - cluster = cluster the vehicle is considering doing the action at
        - eta = Estimated time of arrival for the vehicle to arrive at cluster
        """
        if isinstance(location, sim.Station):
            target_state = round(location.get_target_state(simul.day(),simul.hour())) 
            time_until_arrival = eta - simul.time

            # Calculate number of bikes arriving/leaving in the time period until vehicle arrival
            net_demand = calculate_net_demand(location, simul.time, simul.day(), simul.hour(), min(time_until_arrival, 60))
            max_num_usable_bikes = get_max_num_usable_escooters(location, vehicle)
            max_num_usable_bikes_eta = max_num_usable_bikes + net_demand

            # Calculate the amount of neighbors that are starving or congested, can impact the number of bikes to operate on
            num_starved_neighbors = 0
            num_overflowing_neighbors = 0
            for neighbor in location.get_neighbours():
                net_demand_neighbor = calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(), min(time_until_arrival, 60))
                num_usable_bikes_neighbor_eta = len(neighbor.get_available_bikes()) + net_demand_neighbor
                target_state_neighbor = neighbor.get_target_state(simul.day(), simul.hour())
                if num_usable_bikes_neighbor_eta < self.starvation_criteria * target_state_neighbor:
                    num_starved_neighbors += 1
                elif num_usable_bikes_neighbor_eta > self.overflow_criteria * target_state_neighbor:
                    num_overflowing_neighbors += 1

            number_of_bikes_pickup = 0
            number_of_bikes_deliver = 0
            number_of_battery_swaps = 0

            # Calculate how many bikes to do different actions on
            if max_num_usable_bikes_eta < target_state: # deliver
                number_of_bikes_deliver = min(len(vehicle.get_sb_bike_inventory()), 
                                            target_state - max_num_usable_bikes_eta + BIKES_STARVED_NEIGHBOR * num_starved_neighbors)
                number_of_battery_swaps = min( len(location.get_swappable_bikes()), 
                                            vehicle.battery_inventory)
            
            elif max_num_usable_bikes_eta > target_state: # pick-up
                remaining_cap_vehicle = vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory())
                num_less_bikes = min(remaining_cap_vehicle, 
                                    max_num_usable_bikes_eta - target_state + BIKES_OVERFLOW_NEIGHBOR * num_overflowing_neighbors)

                num_bikes_to_swap_and_pickup = min(len(location.get_swappable_bikes()), 
                                                vehicle.battery_inventory, 
                                                num_less_bikes)
                number_of_bikes_pickup = round(num_less_bikes) - num_bikes_to_swap_and_pickup
                number_of_battery_swaps = min(vehicle.battery_inventory, max(0,len(location.get_swappable_bikes()) - num_bikes_to_swap_and_pickup))
            
            else: # only swap
                swappable_bikes = location.get_swappable_bikes()
                number_of_battery_swaps = min(len(swappable_bikes), vehicle.battery_inventory)
            
            return number_of_bikes_pickup, number_of_bikes_deliver, number_of_battery_swaps
        else:
            target_state = location.get_target_state(simul.day(),simul.hour())
            time_until_arrival = eta - simul.time

            # Calculate number of esooters arriving/leaving in the time period until vehicle arrival
            net_demand = calculate_net_demand(location, simul.time, simul.day(), simul.hour(), min(time_until_arrival, 60))
            max_num_usable_escooters = get_max_num_usable_escooters(location, vehicle)
            max_num_usable_escooters_eta = max_num_usable_escooters + net_demand

            # Calculate the amount of neighbors that are starving or congested, can impact the number of bikes to operate on
            num_usable_bikes_neighbours_eta = 0
            target_neighbours = 0
            for neighbor in location.get_neighbours():
                net_demand_neighbor = calculate_net_demand(neighbor, simul.time, simul.day(), simul.hour(), min(time_until_arrival, 60))
                num_usable_bikes_neighbours_eta += len(neighbor.get_available_bikes()) + net_demand_neighbor
                target_neighbours += neighbor.get_target_state(simul.day(), simul.hour())
            difference_from_target_neighbours = target_neighbours - num_usable_bikes_neighbours_eta # Difference from target state in neighbourhood, negative -> too few bikes, positive -> too many

            number_of_escooters_pickup = 0
            number_of_escooters_deliver = 0
            number_of_escooters_swap = 0

            difference_from_target_here = target_state - max_num_usable_escooters_eta
            neighborhood_difference_target = difference_from_target_here + difference_from_target_neighbours

            # Calculate how many escooters to do different actions on
            #TODO dobbelt sjekk
            if difference_from_target_neighbours < 0: # delivery
                num_escooters = min(len(vehicle.get_ff_bike_inventory()), 
                                    -neighborhood_difference_target)
                escooters_to_deliver_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = get_escooter_ids_load_swap(location, vehicle, num_escooters, "deliver", self.swap_threshold)
                number_of_escooters_deliver = len(escooters_to_deliver_accounted_for_battery_swaps)
                number_of_escooters_swap = len(escooters_to_swap_accounted_for_battery_swap)
            
            elif difference_from_target_neighbours > 0: # pick-up
                remaining_cap_vehicle = vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory())
                number_of_less_escooters = min(remaining_cap_vehicle, 
                                            neighborhood_difference_target,
                                            max_num_usable_escooters_eta)
                escooters_to_pickup_accounted_for_battery_swaps, escooters_to_swap_accounted_for_battery_swap = get_escooter_ids_load_swap(location, vehicle, number_of_less_escooters, "pickup", self.swap_threshold)
                number_of_escooters_pickup = len(escooters_to_pickup_accounted_for_battery_swaps)
                number_of_escooters_swap = len(escooters_to_swap_accounted_for_battery_swap)
            
            else: # only swap
                escooters_in_cluster_low_battery = location.get_swappable_bikes()
                number_of_escooters_swap = min(len(escooters_in_cluster_low_battery), 
                                            vehicle.battery_inventory)
            
            return number_of_escooters_pickup, number_of_escooters_deliver, number_of_escooters_swap
        

    def greedy_next_visit(self, plan, simul, number_of_successors, weight_set, total_num_bikes_in_system):
        visits = []
        tabu_list = plan.tabu_list
        vehicle = plan.next_visit.vehicle

        num_sb_bikes_now = len(vehicle.get_sb_bike_inventory())
        num_ff_bikes_now = len(vehicle.get_ff_bike_inventory())

        # Update the vehicle bike inventory based on the planned operational actions
        for visit in plan.plan[vehicle.vehicle_id]:
            if isinstance(visit.station, sim.Station):
                num_sb_bikes_now += visit.loading_quantity
                num_sb_bikes_now -= visit.unloading_quantity
            else:
                num_ff_bikes_now += visit.loading_quantity
                num_ff_bikes_now -= visit.unloading_quantity

        # Finds potential next stations based on pick up or delivery status of the station and tabulist
        potential_stations = find_potential_stations(simul,0.15,0.15,vehicle, num_sb_bikes_now, num_ff_bikes_now+num_sb_bikes_now, tabu_list) #Hvor kommer 0.15 fra??
        potential_clusters = find_potential_clusters(simul, 0.15, vehicle, num_ff_bikes_now, num_ff_bikes_now+num_sb_bikes_now)
        if potential_stations == [] and potential_clusters == []: #Try to find out when and if this happens?
            return None
        
        number_of_successors = min(number_of_successors, len(potential_stations))

        # Finds the criticality score of all potential locations, and sort them in descending order
        stations_sorted = calculate_criticality_sb(weight_set[1], simul, potential_stations, plan.plan[vehicle.vehicle_id][-1].station, total_num_bikes_in_system, tabu_list) if potential_stations != [] else {}
        clusters_sorted = calculate_criticality_ff(weight_set[0], simul, potential_clusters, plan.plan[vehicle.vehicle_id][-1].station, total_num_bikes_in_system, tabu_list) if potential_clusters != [] else {}
        sorted_criticalities_merged = dict(sorted({**stations_sorted, **clusters_sorted}.items(), key=lambda item: item[1], reverse=True))
        destinations_sorted_list = list(sorted_criticalities_merged.keys())
        
        # Selects the most critical clusters as next visits
        next_destination = [destinations_sorted_list[i] for i in range(number_of_successors)]

        for next in next_destination:
            arrival_time = plan.plan[vehicle.vehicle_id][-1].get_depature_time() + simul.state.get_vehicle_travel_time(plan.plan[vehicle.vehicle_id][-1].station.location_id, next.location_id) + MINUTES_CONSTANT_PER_ACTION
            number_of_escooters_to_pickup, number_of_escooters_to_deliver, number_of_escooters_to_swap = self.calculate_loading_quantities_and_swaps_pilot(vehicle, simul, next, arrival_time)
            new_visit = Visit(next, number_of_escooters_to_pickup, number_of_escooters_to_deliver, number_of_escooters_to_swap, arrival_time, vehicle)
            visits.append(new_visit)
        
        return visits
    
    def return_best_move_avarage(self, vehicle, simul, plan_scores):
        score_board = dict() #plan object: the average score of this plan
        num_scenarios=self.number_of_scenarios
        if num_scenarios==0:
            num_scenarios+=1 #this scenario is now the expected value 
        for scenario_id in range(num_scenarios):
            for plan in plan_scores:
                score = plan_scores[plan][scenario_id]
                if plan in score_board:
                    score_board[plan] += score
                else:
                    score_board[plan] = score

        score_board_sorted = dict(sorted(score_board.items(), key=lambda item: item[1], reverse=True))
        if list(score_board_sorted.keys())[0] != None:
            best_plan = list(score_board_sorted.keys())[0]
            branch = best_plan.branch_number
            simul.metrics.add_aggregate_metric(simul, "branch"+str(branch+1), 1)
            first_move = best_plan.plan[vehicle.vehicle_id][1].station.location_id
            destination = best_plan.plan[vehicle.vehicle_id][1].station
            if isinstance(destination, sim.Station):
                return first_move, None
            return first_move , destination
        else: 
            tabu_list = [vehicle2.location.location_id for vehicle2 in simul.state.get_vehicles()]
            potential_stations2 = [station for station in simul.state.get_areas() if station.location_id not in tabu_list]    
            rng_balanced = np.random.default_rng(None)
            destination = rng_balanced.choice(potential_stations2)
            if isinstance(destination, sim.Station):
                return destination.location_id, None
            return destination.location_id, destination

def calculate_loading_quantities_and_swaps_greedy(vehicle, simul, location, overflow_criteria, starvation_criteria, swap_threshold):
    """
    Returns a list of IDs of the bikes to deliver, pickup or swap batteries on.
    The calculation is done when a vehicle arrives at the station, and the list returned are performed.

    Parameters:
    - vehicle = Vehicle-object that is doing the action
    - simul = Simulator
    - location = The Area or Cluster under consideration
    - congestion_criteria = percentage of station capacity for a station to be considered congested
    - starvation_critera = percentage of station capacity for a station to be considered starved
    """
    target_state = round(location.get_target_state(simul.day(), simul.hour())) #Denne må vi finne ut hvordan lages
    num_max_usable_bikes_after_visit = get_max_num_usable_escooters(location, vehicle)

    # Count how many neighbors are starved or congested
    starved_neighbors = 0
    overflowing_neighbors = 0
    for neighbor in location.get_neighbours():
        num_bikes_neighbor = len(neighbor.get_available_bikes())
        neighbor_target_state = neighbor.get_target_state(simul.day(), simul.hour())
        if num_bikes_neighbor < starvation_criteria * neighbor_target_state:
            starved_neighbors += 1
        elif num_bikes_neighbor > overflow_criteria * neighbor_target_state:
            overflowing_neighbors += 1
        #TODO legge til congestion?

    # If the location is a delivery, calculate which bikes to deliver from the vehicle, and which bikes to swap on
    if num_max_usable_bikes_after_visit < target_state:
        if isinstance(location, sim.Station):
            num_bikes_to_deliver = min(len([bike for bike in vehicle.get_sb_bike_inventory() if bike.usable()]), 
                                                 target_state - num_max_usable_bikes_after_visit + BIKES_STARVED_NEIGHBOR * starved_neighbors
                                                 )
        else:
            num_bikes_to_deliver = min(len([escooter for escooter in vehicle.get_ff_bike_inventory() if escooter.usable()]), 
                                                 target_state - num_max_usable_bikes_after_visit + BIKES_STARVED_NEIGHBOR * starved_neighbors
                                                 )
        bikes_to_deliver, bikes_to_swap = get_escooter_ids_load_swap(location, vehicle, num_bikes_to_deliver, "deliver", swap_threshold)
        bikes_to_pickup = []

    # If the location is a pickup, calculate which to pickup, and which to swap batteries on
    elif num_max_usable_bikes_after_visit > target_state:
        remaining_vehicle_capacity = vehicle.bike_inventory_capacity - len(vehicle.get_bike_inventory())
        number_of_escooters_to_pickup = min(remaining_vehicle_capacity, 
                                            num_max_usable_bikes_after_visit - target_state + BIKES_OVERFLOW_NEIGHBOR * overflowing_neighbors, 
                                            len(location.get_bikes()))
        bikes_to_pickup, bikes_to_swap = get_escooter_ids_load_swap(location, vehicle, number_of_escooters_to_pickup, "pickup", swap_threshold)
        bikes_to_deliver=[]

    # If no bikes need to be picked up or delivered, find out how many bikes to swap batteries on
    else:
        bikes_to_pickup = []
        bikes_to_deliver = []
        swappable_bikes_at_location = location.get_swappable_bikes()
        num_bikes_to_swap = min(len(swappable_bikes_at_location),vehicle.battery_inventory)
        bikes_to_swap = [bike.bike_id for bike in swappable_bikes_at_location[:num_bikes_to_swap]]
    
    return bikes_to_pickup, bikes_to_deliver, bikes_to_swap

def get_escooter_ids_load_swap(location, vehicle, num_bikes, location_type, swap_threshold):
    """
    Returns lists of the IDs of the bikes to deliver/pick-up and swap.

    Parameters:
    - station = Station being considered
    - vehicle = Vehicle doing the action
    - num_bikes = difference from target state after battery swap on site is done + effects of neighbors
    - station_type = if there has to be unloading or pick-ups done at the station
    """
    if isinstance(location, sim.Area):
        return get_escooter_ids_load_swap_ff(location, vehicle, num_bikes, location_type, swap_threshold)
    else:
        return get_bike_ids_load_swap(location, vehicle, num_bikes, location_type)
    
def find_potential_clusters(simul, cutoff_vehicle, vehicle, ff_bikes_in_vehicle, bikes_in_vehicle):
    """
    Returns a list of Station-Objects that are not in the tabu list, and that need help to reach target state.

    Parameters:
    - simul = Simulator
    - cutoff_vehicle = At what percentage of the vehicle's capacity is the vehicle considered "empty" or "full"
    - vehicle = Vehicle-object under consideration
    - bikes_at_vehicle = number of bikes in the vehicle at the time
    """
    potential_pickup_clusters = []
    potential_delivery_clusters = []

    # Vehicle's inventory is capable of both pickups and deliveries
    if cutoff_vehicle * vehicle.bike_inventory_capacity <= bikes_in_vehicle <= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
        potential_pickup_clusters = clusterPickup(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, MAX_WALKING_AREAS, vehicle, simul)
        if ff_bikes_in_vehicle >= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
            potential_delivery_clusters = clusterDelivery(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, MAX_WALKING_AREAS, vehicle, simul)

        potential_stations = potential_pickup_clusters + potential_delivery_clusters
    else:
        potential_stations = []
        # Vehicle's inventory is not able to deliver escooters
        if bikes_in_vehicle <= cutoff_vehicle*vehicle.bike_inventory_capacity:
            potential_stations = clusterPickup(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, MAX_WALKING_AREAS, vehicle, simul)
        # Vehicle's inventory is not able to pick up more escooters
        elif ff_bikes_in_vehicle >= (1-cutoff_vehicle)*vehicle.bike_inventory_capacity:
            potential_stations = clusterDelivery(simul.state.get_areas(), MAX_NUMBER_OF_CLUSTERS, MAX_WALKING_AREAS, vehicle, simul)

    return potential_stations

def find_potential_stations(simul, cutoff_vehicle, cutoff_station, vehicle, sb_bikes_in_vehicle, bikes_in_vehicle, tabu_list):
    """
    Returns a list of Station-Objects that are not in the tabu list, and that need help to reach target state.

    Parameters:
    - simul = Simulator
    - cutoff_vehicle = At what percentage is the vehicle considered "empty" or "full"
    - cutoff_station = At what percentage above or below the target state is a station considered a pickup or delivery
    - vehicle = Vehicle-object under consideration
    - bikes_at_vehicle = number of bikes in the vehicle at the time
    - tabu_list = list of Station-objects that are currently being visited or soon to be
    """
    # Filter out stations in tabulist
    potential_stations = [station for station in simul.state.get_stations() if station.location_id not in tabu_list]
    
    # Makes dictionary with net_demand and target state for all possible stations
    net_demands = {station.location_id: calculate_net_demand(station, simul.time, simul.day(), simul.hour(), 60) for station in potential_stations}
    target_states = {station.location_id: station.get_target_state(simul.day(), simul.hour()) for station in potential_stations}
    
    # If the available bikes in the future is bigger that a cutoff percentage of target state, the station is a pickup station
    potential_pickup_stations = [station for station in potential_stations
                                 if get_max_num_usable_escooters(station, vehicle) + net_demands[station.location_id] > (1 + cutoff_station) * target_states[station.location_id]
                                 ] # TODO blir cuttoff annerledes med station based og capacity?
    
    # If the available bikes after a visit in the future is lower that a cutoff percentage of target state, the station is a delivery station
    potential_delivery_stations = [ station for station in potential_stations
                                 if get_max_num_usable_escooters(station, vehicle) + net_demands[station.location_id] < (1 - cutoff_station) * target_states[station.location_id]
                                 ]
    
    #Decides whether pickup, delivery or balanced is relevant
    if cutoff_vehicle * vehicle.bike_inventory_capacity <= bikes_in_vehicle <= (1 - cutoff_vehicle) * vehicle.bike_inventory_capacity:
        if bikes_in_vehicle >= (1 - cutoff_vehicle) * vehicle.bike_inventory_capacity:
            potential_stations = potential_delivery_stations
        potential_stations += potential_pickup_stations
    else:
        if bikes_in_vehicle <= cutoff_vehicle * vehicle.bike_inventory_capacity:
            potential_stations = potential_pickup_stations

        elif bikes_in_vehicle >= (1 - cutoff_vehicle) * vehicle.bike_inventory_capacity:
            potential_stations = potential_delivery_stations
    
    return potential_stations