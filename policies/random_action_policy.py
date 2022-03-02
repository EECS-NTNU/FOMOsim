"""
This file contains all the policies used in the thesis.
"""
import copy
import math

from policies import Policy, neighbour_filtering
import sim
import abc
from sim import State, Vehicle

def get_possible_actions(
    state,
    vehicle: Vehicle,
    number_of_neighbours=None,
    divide=None,
    exclude=None,
    time=None,
):
    """
    Enumerate all possible actions from the current state
    :param time: time of the world when the actions is to be performed
    :param exclude: clusters to exclude from next cluster
    :param vehicle: vehicle to perform this action
    :param number_of_neighbours: number of neighbours to evaluate, if None: all neighbors are returned
    :param divide: number to divide by to create range increment
    :return: List of Action objects
    """
    actions = []
    if number_of_neighbours is None:
        neighbours = state.stations
    else:
        neighbours = neighbour_filtering.filtering_neighbours(
            state,
            vehicle,
            0,
            0,
            number_of_neighbours,
            exclude=exclude,
        )
    # Return empty action if
    if not vehicle.is_at_depot():

        def get_range(max_int):
            if divide and divide > 0 and max_int > 0:
                return list(
                    {
                        *(
                            [
                                i
                                for i in range(
                                    0, max_int + 1, math.ceil(max_int / divide)
                                )
                            ]
                            + [max_int]
                        )
                    }
                )
            else:
                return [i for i in range(max_int + 1)]

        # Initiate constraints for battery swap, pick-up and drop-off
        pick_ups = min(
            max(
                len(vehicle.current_location.scooters),
                0,
            ),
            vehicle.scooter_inventory_capacity - len(vehicle.scooter_inventory),
            vehicle.battery_inventory,
        )
        swaps = vehicle.get_max_number_of_swaps()
        drop_offs = len(vehicle.scooter_inventory)
        combinations = []
        # Different combinations of battery swaps, pick-ups, drop-offs and clusters
        for pick_up in get_range(pick_ups):
            for swap in get_range(swaps):
                for drop_off in get_range(drop_offs):
                    if (
                        (pick_up + swap) <= len(vehicle.current_location.scooters)
                        and (pick_up + swap) <= vehicle.battery_inventory
                        and (pick_up + swap + drop_off > 0)
                    ):

                        ngbrs = state.stations
                        if number_of_neighbours is not None:
                            ngbrs = neighbour_filtering.filtering_neighbours(
                                state,
                                vehicle,
                                pick_up,
                                drop_off,
                                number_of_neighbours,
                                exclude=exclude,
                            )

                        for location in ngbrs:
                            combinations.append(
                                [
                                    max(
                                        min(
                                            vehicle.battery_inventory - pick_up,
                                            swap,
                                        ),
                                        0,
                                    ),
                                    pick_up,
                                    drop_off,
                                    location.id,
                                ]
                            )

        # Assume that no battery swap or pick-up of scooters with 100% battery and
        # that the scooters with the lowest battery are prioritized
        swappable_scooters_id = [
            scooter.id
            for scooter in vehicle.current_location.get_swappable_scooters()
        ]

        none_swappable_scooters_id = [
            scooter.id
            for scooter in vehicle.current_location.scooters
            if isinstance(scooter, sim.Bike) or scooter.battery >= 70
        ]

        def choose_pick_up(swaps, pickups):
            number_of_none_swappable_scooters = max(
                swaps + pickups - len(swappable_scooters_id), 0
            )

            return (
                swappable_scooters_id[swaps : swaps + pick_up]
                + none_swappable_scooters_id[:number_of_none_swappable_scooters]
            )

        # Adding every action. Actions are the IDs of the scooters to be handled.
        for battery_swap, pick_up, drop_off, cluster_id in combinations:
            if not vehicle.is_at_depot() and (
                vehicle.battery_inventory - battery_swap - pick_up
                < vehicle.battery_inventory_capacity * 0.1
            ):
                # If battery inventory is low, go to depot
                cluster_id = [
                    depot.id
                    for depot in sorted(
                        state.depots,
                        key=lambda depot: state.get_distance(
                            vehicle.current_location.id, depot.id
                        ),
                    )
                    if depot.get_available_battery_swaps(time)
                    > vehicle.battery_inventory_capacity * 0.9
                ][0]
            actions.append(
                sim.Action(
                    swappable_scooters_id[:battery_swap],
                    choose_pick_up(battery_swap, pick_up),
                    [scooter.id for scooter in vehicle.scooter_inventory][
                        :drop_off
                    ],
                    cluster_id,
                )
            )
    return (
        actions
        if len(actions) > 0
        else [sim.Action([], [], [], neighbour.id) for neighbour in neighbours]
    )

class RandomActionPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_best_action(self, simul, vehicle):
        # all possible actions in this state
        possible_actions = get_possible_actions(
            simul.state,
            vehicle,
            time=simul.time,
            # divide=2,
            # number_of_neighbours=2,
        )

        # pick a random action
        return simul.state.rng.choice(possible_actions)
