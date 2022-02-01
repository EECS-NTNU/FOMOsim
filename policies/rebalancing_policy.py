"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import classes
import numpy.random as random
import abc

import system_simulation.scripts

class RebalancingPolicy(Policy):
    def __init__(self):
        super().__init__(0, 0)

    def get_best_action(self, world, vehicle):
        vehicle_has_scooter_inventory = len(vehicle.scooter_inventory) > 0
        if vehicle.is_at_depot():
            scooters_to_deliver = []
            scooters_to_pickup = []
            number_of_scooters_to_pick_up = 0
            number_of_scooters_to_swap = 0
            scooters_to_swap = []
        else:
            # If vehicle has scooter inventory, deliver all scooters and swap all swappable scooters
            if vehicle_has_scooter_inventory:
                # Deliver all scooters in scooter inventory, and don't pick up any new scooters
                scooters_to_deliver = [
                    scooter.id for scooter in vehicle.scooter_inventory
                ]
                scooters_to_pickup = []
                number_of_scooters_to_pick_up = 0
                # Swap as many scooters as possible as this cluster most likely needs it
                swappable_scooters = vehicle.current_location.get_swappable_scooters()
                number_of_scooters_to_swap = min(
                    vehicle.battery_inventory, len(swappable_scooters)
                )
                scooters_to_swap = [scooter.id for scooter in swappable_scooters][
                    :number_of_scooters_to_swap
                ]
            else:
                # Pick up as many scooters as possible, the min(scooter capacity, deviation from ideal state)
                number_of_scooters_to_pick_up = max(
                    min(
                        vehicle.scooter_inventory_capacity
                        - len(vehicle.scooter_inventory),
                        vehicle.battery_inventory,
                        len(vehicle.current_location.scooters)
                        - vehicle.current_location.ideal_state,
                    ),
                    0,
                )
                scooters_to_pickup = [
                    scooter.id for scooter in vehicle.current_location.scooters
                ][:int(number_of_scooters_to_pick_up)]
                # Do not swap any scooters in a cluster with a lot of scooters
                scooters_to_swap = []
                number_of_scooters_to_swap = 0
                # There are no scooters to deliver due to empty inventory
                scooters_to_deliver = []

        def get_next_location_id(is_finding_positive_deviation):
            return sorted(
                [
                    cluster
                    for cluster in world.state.clusters
                    if cluster.id != vehicle.current_location.id
                    and cluster.id not in world.tabu_list
                ],
                key=lambda cluster: len(cluster.get_available_scooters())
                - cluster.ideal_state,
                reverse=is_finding_positive_deviation,
            )[0].id

        # If vehicles has under 10% battery inventory, go to depot.
        if (
            vehicle.battery_inventory
            - number_of_scooters_to_swap
            - number_of_scooters_to_pick_up
            < vehicle.battery_inventory_capacity * 0.1
        ) and not vehicle.is_at_depot():
            next_location_id = world.state.depots[0].id
        else:
            """
            If vehicle has scooter inventory upon arrival,
            go to new positive deviation cluster to pick up new scooters.
            If there are no scooter inventory, go to cluster where you
            can drop off scooters picked up in this cluster, ergo negative deviation cluster.
            If, however, you are in the depot, you should do the opposite as the depot does not
            change the scooter inventory.
            """
            visit_positive_deviation_cluster_next = (
                vehicle_has_scooter_inventory
                if not vehicle.is_at_depot()
                else not vehicle_has_scooter_inventory
            )
            next_location_id = get_next_location_id(
                visit_positive_deviation_cluster_next
            )

        return classes.Action(
            scooters_to_swap,
            scooters_to_pickup,
            scooters_to_deliver,
            next_location_id,
        )
