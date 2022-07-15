"""
This file contains a greedy policy
"""
import copy

from policies import Policy
import sim
# import abc

# import init_state
import init_state.entur.methods
import init_state.entur.scripts

class GreedyPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_best_action(self, simul, vehicle):
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
                    scooter.id for scooter in vehicle.get_scooter_inventory()
                ]
                scooters_to_pickup = []
                number_of_scooters_to_pick_up = 0
                # Swap as many scooters as possible as this station most likely needs it
                swappable_scooters = vehicle.current_location.get_swappable_scooters()
                number_of_scooters_to_swap = min(
                    vehicle.battery_inventory, len(swappable_scooters)
                )
                scooters_to_swap = [scooter.id for scooter in swappable_scooters][
                    :number_of_scooters_to_swap
                ]
            else:
                # Pick up as many scooters as possible, the min(scooter capacity, deviation from ideal state)
                number_of_scooters_to_pick_up = int(max(
                    min(
                        vehicle.scooter_inventory_capacity
                        - len(vehicle.scooter_inventory),
                        vehicle.battery_inventory,
                        len(vehicle.current_location.scooters)
                        - vehicle.current_location.get_target_state(simul.day(), simul.hour()),
                    ),
                    0,
                ))
                scooters_to_pickup = list(vehicle.current_location.scooters.keys())[:number_of_scooters_to_pick_up]
                # Do not swap any scooters in a station with a lot of scooters
                scooters_to_swap = []
                number_of_scooters_to_swap = 0
                # There are no scooters to deliver due to empty inventory
                scooters_to_deliver = []

        def get_next_location_id(simul, is_finding_positive_deviation):
            tabu_list = [ vehicle.current_location.id for vehicle in simul.state.vehicles ]

            return sorted(
                [
                    station
                    for station in simul.state.stations.values()
                    if station.id not in tabu_list
                ],
                key=lambda station: len(station.get_available_scooters())
                - station.get_target_state(simul.day(), simul.hour()),
                reverse=is_finding_positive_deviation,
            )[0].id

        # If vehicles has under 10% battery inventory, go to depot.
        if (
            vehicle.battery_inventory
            - number_of_scooters_to_swap
            - number_of_scooters_to_pick_up
            < vehicle.battery_inventory_capacity * 0.1
        ) and not vehicle.is_at_depot() and (len(simul.state.depots) > 0):
            next_location_id = list(simul.state.depots.values())[0].id
        else:
            """
            If vehicle has scooter inventory upon arrival,
            go to new positive deviation station to pick up new scooters.
            If there are no scooter inventory, go to station where you
            can drop off scooters picked up in this station, ergo negative deviation station.
            If, however, you are in the depot, you should do the opposite as the depot does not
            change the scooter inventory.
            """
            visit_positive_deviation_station_next = (len(vehicle.scooter_inventory) + len(scooters_to_pickup) - len(scooters_to_deliver)) <= 0
            next_location_id = get_next_location_id(simul, visit_positive_deviation_station_next)

        return sim.Action(
            scooters_to_swap,
            scooters_to_pickup,
            scooters_to_deliver,
            next_location_id,
        )
