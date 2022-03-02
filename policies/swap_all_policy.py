"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import abc

def get_current_state(station) -> float:
    return sum(map(lambda scooter: 1 if isinstance(scooter, sim.Bike) else scooter.battery / 100, station.scooters))

class SwapAllPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_best_action(self, simul, vehicle):
        # Choose a random cluster

        if vehicle.is_at_depot():
            swappable_scooters_ids = []
            number_of_scooters_to_swap = 0
        else:
            # Find all scooters that can be swapped here
            swappable_scooters_ids = [
                scooter.id
                for scooter in vehicle.current_location.get_swappable_scooters(
                    battery_limit=70
                )
            ]

            # Calculate how many scooters that can be swapped
            number_of_scooters_to_swap = vehicle.get_max_number_of_swaps()

        if (
            vehicle.battery_inventory - number_of_scooters_to_swap
            < vehicle.battery_inventory_capacity * 0.1
        ) and not vehicle.is_at_depot():
            next_location = simul.state.depots[0]
        else:
            next_location = sorted(
                [
                    cluster
                    for cluster in simul.state.stations
                    if cluster.id != vehicle.current_location.id
                ],
                key=lambda cluster: (
                    len(cluster.scooters) - get_current_state(cluster)
                )
                / (len(cluster.scooters) + 1),
                reverse=True,
            )[0]

        # Return an action with no re-balancing, only scooter swapping
        return sim.Action(
            battery_swaps=swappable_scooters_ids[:number_of_scooters_to_swap],
            pick_ups=[],
            delivery_scooters=[],
            next_location=next_location.id,
        )

    def initSim(self, sim):
        for vehicle in sim.state.vehicles:
            vehicle.battery_inventory_capacity = 250
            vehicle.battery_inventory = 250
            vehicle.scooter_inventory_capacity = 0
            vehicle.scooter_inventory = []
