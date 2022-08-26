"""
This file contains all the policies used in the thesis.
"""
import copy
import os

from progress.bar import Bar
import sim
import numpy as np
import abc
import math

from policies import Policy
import policies.haflan_haga_spetalen.settings as epssettings
import policies.neighbour_filtering
import policies.haflan_haga_spetalen.system_simulation.scripts
import settings
import init_state.entur
import init_state.entur.methods
import init_state.entur.scripts
from policies.haflan_haga_spetalen.helpers import *

def get_current_state(station) -> float:
    return sum(map(lambda scooter: 1 if isinstance(scooter, sim.Bike) else scooter.battery / 100, station.get_scooters()))


def get_possible_actions(
    state, day, hour,
    vehicle: sim.Vehicle,
    number_of_neighbours,
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
    neighbours = policies.neighbour_filtering.filtering_neighbours(
        state, day, hour, 
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
                            + [max_int]
                        )
                    }
                )
            else:
                return [i for i in range(int(max_int + 1))]

        # Initiate constraints for battery swap, pick-up and drop-off
        pick_ups = min(
            max(
                len(vehicle.location.bikes)
                - vehicle.location.get_target_state(day, hour),
                0,
            ),
            vehicle.bike_inventory_capacity - len(vehicle.bike_inventory),
            vehicle.battery_inventory,
        )
        swaps = vehicle.get_max_number_of_swaps()
        drop_offs = max(
            min(
                vehicle.location.get_target_state(day, hour)
                - len(vehicle.location.bikes),
                len(vehicle.bike_inventory),
            ),
            0,
        )
        combinations = []
        # Different combinations of battery swaps, pick-ups, drop-offs and clusters
        for pick_up in get_range(pick_ups):
            for swap in get_range(swaps):
                for drop_off in get_range(drop_offs):
                    if (
                        (pick_up + swap) <= len(vehicle.location.bikes)
                        and (pick_up + swap) <= vehicle.battery_inventory
                        and (pick_up + swap + drop_off > 0)
                    ):
                        for (
                            location
                        ) in policies.neighbour_filtering.filtering_neighbours(
                            state, day, hour, 
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
            for scooter in vehicle.location.get_swappable_bikes()
        ]

        none_swappable_scooters_id = [
            scooter.id
            for scooter in vehicle.location.bikes.values()
            if isinstance(scooter, sim.Bike) or (scooter.battery >= 70)
        ]

        def choose_pick_up(swaps, pickups):
            number_of_none_swappable_scooters = max(
                swaps + pickups - len(swappable_scooters_id), 0
            )

            return (
                swappable_scooters_id[int(swaps) : int(swaps + pick_up)]
                + none_swappable_scooters_id[:int(number_of_none_swappable_scooters)]
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
                        state.depots.values(),
                        key=lambda depot: state.get_travel_time(
                            vehicle.location.id, depot.id
                        ),
                    )
                    if depot.get_available_battery_swaps(time)
                    > vehicle.battery_inventory_capacity * 0.9
                ][0]
            actions.append(
                sim.Action(
                    swappable_scooters_id[:int(battery_swap)],
                    choose_pick_up(battery_swap, pick_up),
                    [scooter.id for scooter in vehicle.get_bike_inventory()][
                        :int(drop_off)
                    ],
                    cluster_id,
                )
            )
    return (
        actions
        if len(actions) > 0
        else [sim.Action([], [], [], neighbour.id) for neighbour in neighbours]
    )

class EpsilonGreedyValueFunctionPolicy(Policy):
    """
    This is the primary policy of the thesis. It chooses an action based on a epsilon greedy policy.
    Will update weights after chosen action
    """

    def __init__(
        self,
        get_possible_actions_divide,
        number_of_neighbors,
        epsilon,
        value_function,
    ):
        super().__init__()
        self.get_possible_actions_divide = get_possible_actions_divide
        self.number_of_neighbors = number_of_neighbors
        self.value_function = value_function
        self.epsilon = epsilon
        self.decision_times = []

    @staticmethod
    def get_cache(state):
        # Cache current states in state
        current_states, available_scooters = [], []
        for cluster in state.stations.values():
            current_states.append(get_current_state(cluster))
            available_scooters.append(cluster.get_available_bikes())
        return current_states, available_scooters

    def get_best_action(self, simul, vehicle):
        tabu_list = [ vehicle.location.id for vehicle in simul.state.vehicles ]

        # Find all possible actions
        actions = get_possible_actions(
            simul.state, simul.day(), simul.hour(), 
            vehicle,
            divide=self.get_possible_actions_divide,
            exclude=tabu_list,
            time=simul.time,
            number_of_neighbours=self.number_of_neighbors,
        )
        state = simul.state
        cache = EpsilonGreedyValueFunctionPolicy.get_cache(state)
        # Get state representation of current state
        state_features = self.value_function.get_state_features(
            simul.state, simul.day(), simul.hour(), vehicle, cache
        )

        # Epsilon greedy choose an action based on value function
        if self.epsilon > simul.state.rng.random():
            best_action = simul.state.rng.choice(actions)
        else:
            # Create list containing all actions and their rewards and values (action, reward, value_function_value)
            action_info = [
                (
                    sim.Action([], [], [], simul.state.rng.choice(simul.state.locations).id),
                    -1000,
                    [],
                )  # No actions bug
            ]
            reward = 0
            for action in actions:
                # look one action ahead
                forward_state: sim.State = state.sloppycopy()
                forward_state.simulation_scenarios = state.simulation_scenarios
                forward_vehicle: sim.Vehicle = forward_state.get_vehicle_by_id(
                    vehicle.id
                )
                # perform action
                forward_state.do_action(action, forward_vehicle, simul.time)
                # Simulate the system to generate potential lost trips
                _, _, lost_demands = policies.haflan_haga_spetalen.system_simulation.scripts.system_simulate(
                    forward_state, simul.day(), simul.hour()
                )
                # Record lost trip rewards
                reward = (
                    sum(map(lambda lost_trips: lost_trips[0], lost_demands))
                    if len(lost_demands) > 0
                    else 0
                )
                # Find all actions after taking the action moving the state to s_{t+1}
                next_action_actions = get_possible_actions(forward_state, simul.day(), simul.hour(),
                    forward_vehicle, 
                    divide=self.get_possible_actions_divide,
                    exclude=tabu_list + [action.next_location],
                    time=simul.time
                    + action.get_action_time(
                        state.get_travel_time(
                            vehicle.location.id,
                            forward_vehicle.location.id,
                        )
                    ),
                    number_of_neighbours=self.number_of_neighbors,
                )
                cache = EpsilonGreedyValueFunctionPolicy.get_cache(forward_state)
                forward_action_info = []
                for next_state_action in next_action_actions:
                    # Generate the features for this new state after the action
                    next_state_features = self.value_function.get_next_state_features(
                        forward_state,
                        simul.day(), simul.hour(), 
                        forward_vehicle,
                        next_state_action,
                        cache,
                    )
                    # Calculate the expected future reward of being in this new state
                    next_state_value = (
                        self.value_function.estimate_value_from_state_features(
                            next_state_features
                        )
                    )
                    # Add the transition to a list for later evaluation
                    forward_action_info.append(
                        (next_state_action, next_state_value, next_state_features)
                    )

                # find the greedy best next action
                best_next_action, next_state_value, next_state_features = max(
                    forward_action_info, key=lambda pair: pair[1]
                )
                # Add this transition for later evaluation
                action_info.append(
                    (
                        action,
                        next_state_value + reward * epssettings.LOST_TRIP_REWARD,
                        next_state_features,
                    )
                )
            # Choose the action with the highest value and reward
            best_action, next_state_value, next_state_features = max(
                action_info, key=lambda pair: pair[1]
            )
            if not hasattr(simul, "disable_training"):
                simul.disable_training = False
            if not simul.disable_training:
                if self.value_function.use_replay_buffer():
                    self.value_function.train(simul.state.rng, epssettings.REPLAY_BUFFER_SIZE)
                else:
                    self.value_function.train(simul.state.rng, 
                        (
                            state_features,
                            reward * epssettings.LOST_TRIP_REWARD,
                            next_state_features,
                        )
                    )
        return best_action, state_features

    def setup_from_state(self, state):
        self.value_function.setup(state)

    def __str__(self):
        return f"EpsilonGreedyPolicy w/ {self.value_function.__str__()}"
