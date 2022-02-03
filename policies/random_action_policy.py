"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy, neighbour_filtering
import sim
import numpy.random as random
import abc
from sim import State, Vehicle

class RandomActionPolicy(Policy):
    def __init__(self, get_possible_actions_divide, number_of_neighbors):
        super().__init__(get_possible_actions_divide, number_of_neighbors)

    def get_possible_actions(
        self,
        state: State,
        vehicle: Vehicle,
        number_of_neighbours,
        divide=None,
        exclude=None,
        time=None,
    ):
        """
        Enumerate all possible actions from the current state
        :param time: time of the world when the actions is to be performed
        :param exclude: stations to exclude from next cluster
        :param vehicle: vehicle to perform this action
        :param number_of_neighbours: number of neighbours to evaluate, if None: all neighbors are returned
        :param divide: number to divide by to create range increment
        :return: List of Action objects
        """
        actions = []
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
                                        0, int(max_int + 1), int(math.ceil(max_int / divide))
                                    )
                                ]
                                + [int(max_int)]
                            )
                        }
                    )
                else:
                    return [i for i in range(int(max_int + 1))]

            # Initiate constraints for battery swap, pick-up and drop-off
            pick_ups = min(
                len(vehicle.current_location.scooters),
                vehicle.scooter_inventory_capacity - len(vehicle.scooter_inventory),
                vehicle.battery_inventory,
            )
            swaps = vehicle.get_max_number_of_swaps()
            drop_offs = max(
                min(
                    len(vehicle.current_location.scooters),
                    len(vehicle.scooter_inventory),
                ),
                0,
            )
            combinations = []
            # Different combinations of battery swaps, pick-ups, drop-offs and stations
            for pick_up in get_range(pick_ups):
                for swap in get_range(swaps):
                    for drop_off in get_range(drop_offs):
                        if (
                            (pick_up + swap) <= len(vehicle.current_location.scooters)
                            and (pick_up + swap) <= vehicle.battery_inventory
                            and (pick_up + swap + drop_off > 0)
                        ):
                            for (
                                location
                            ) in policies.neighbour_filtering.filtering_neighbours(
                                state,
                                vehicle,
                                pick_up,
                                drop_off,
                                number_of_neighbours,
                                exclude=exclude,
                            ):
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
                if scooter.hasBattery() and scooter.battery >= 70
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
                    Action(
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

    def get_best_action(self, world, vehicle):
        # all possible actions in this state
        possible_actions = self.get_possible_actions(
            world.state,
            vehicle,
            exclude=world.tabu_list,
            time=world.time,
            divide=self.get_possible_actions_divide,
            number_of_neighbours=self.number_of_neighbors,
        )

        # pick a random action
        if len(possible_actions) > 0:
          return random.choice(possible_actions)
        else:
          return sim.Action([], [], [], 0)
