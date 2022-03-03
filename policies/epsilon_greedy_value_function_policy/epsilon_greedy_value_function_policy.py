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
import policies.epsilon_greedy_value_function_policy.settings as epssettings
import policies.neighbour_filtering
import policies.epsilon_greedy_value_function_policy.system_simulation.scripts
import settings
import clustering
import clustering.methods
import clustering.scripts
from policies.epsilon_greedy_value_function_policy.helpers import *

def get_current_state(station) -> float:
    return sum(map(lambda scooter: 1 if isinstance(scooter, sim.Bike) else scooter.battery / 100, station.scooters))


def compute_and_set_ideal_state(state: sim.State, sample_scooters: list):
    progressbar = Bar(
        "| Computing ideal state", max=len(os.listdir(settings.TEST_DATA_DIRECTORY))
    )
    number_of_scooters_counter = np.zeros(
        (len(state.locations), len(os.listdir(settings.TEST_DATA_DIRECTORY)))
    )
    for index, file_path in enumerate(sorted(os.listdir(settings.TEST_DATA_DIRECTORY))):
        progressbar.next()
        current_snapshot = clustering.methods.read_bounded_csv_file(f"{settings.TEST_DATA_DIRECTORY}/{file_path}")
        current_snapshot = current_snapshot[
            current_snapshot["id"].isin(sample_scooters)
        ]
        current_snapshot["cluster"] = [
            state.get_cluster_by_lat_lon(row["lat"], row["lon"]).id
            for index, row in current_snapshot.iterrows()
        ]
        for cluster in state.stations:
            number_of_scooters_counter[cluster.id][index] = len(
                current_snapshot[current_snapshot["cluster"] == cluster.id]
            )
    cluster_ideal_states = np.mean(number_of_scooters_counter, axis=1)
    normalized_cluster_ideal_states = normalize_to_integers(
        cluster_ideal_states, sum_to=len(sample_scooters)
    )

    progressbar.finish()

    for cluster in state.stations:
        cluster.ideal_state = normalized_cluster_ideal_states[cluster.id]

    # setting number of scooters to ideal state
    state_rebalanced_ideal_state = idealize_state(state)

    # adjusting ideal state by average cluster in- and outflow
    simulate_state_outcomes(state_rebalanced_ideal_state, state)


def generate_scenarios(state: sim.State, number_of_scenarios=10000):
    """
    Generate system simulation scenarios. This is used to speed up the training simulation
    :param state: new state
    :param number_of_scenarios: how many scenarios to generate
    :return: the scenarios list of (cluster id, number of trips, list of end cluster ids)
    """
    scenarios = []
    cluster_indices = np.arange(len(state.locations))
    for i in range(number_of_scenarios):
        one_scenario = []
        for cluster in state.stations:
            number_of_trips = round(
                state.rng.poisson(cluster.leave_intensity_per_iteration)
            )
            end_cluster_indices = state.rng.choice(
                cluster_indices,
                p=cluster.get_leave_distribution(state),
                size=number_of_trips,
            ).tolist()
            one_scenario.append((cluster.id, number_of_trips, end_cluster_indices))

        scenarios.append(one_scenario)
    return scenarios

def get_initial_state(
    sample_size=None,
    number_of_clusters=20,
    save=True,
    cache=True,
    initial_location_depot=True,
    number_of_vans=4,
) -> sim.State:
    """
    Main method for setting up a state object based on EnTur data from test_data directory.
    This method saves the state objects and reuse them if a function call is done with the same properties.
    :param sample_size: number of e-scooters in the instance
    :param number_of_clusters
    :param save: if the model should save the state
    :param cache: if the model should check for a cached version
    :param initial_location_depot: set the initial current location of the vehicles to the depot
    :param number_of_vans: the number of vans to use
    :return:
    """
    # If this combination has been requested before we fetch a cached version
    filepath = (
        f"{settings.STATE_CACHE_DIR}/"
        f"{sim.State.save_path(number_of_clusters, sample_size)}.pickle"
    )
    if cache and os.path.exists(filepath):
        print(f"\nUsing cached version of state from {filepath}\n")
        initial_state = sim.State.load(filepath)
    else:

        initial_state = clustering.scripts.get_initial_state(
            "Scooter",
            number_of_scooters = sample_size,
            number_of_clusters = number_of_clusters,
            initial_location_depot = initial_location_depot,
            number_of_vans = number_of_vans,
        )

        # Generate scenarios
        initial_state.simulation_scenarios = generate_scenarios(initial_state)

        entur_dataframe = clustering.methods.read_bounded_csv_file(
            "test_data/0900-entur-snapshot.csv"
        )
        sample_scooters = clustering.scripts.scooter_sample_filter(initial_state.rng, entur_dataframe, sample_size)

        # Find the ideal state for each cluster
        compute_and_set_ideal_state(initial_state, sample_scooters)

        if save:
            # Cache the state for later
            initial_state.save_state()
            print("Setup state completed\n")

    return initial_state


def get_possible_actions(
    state,
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
                            + [max_int]
                        )
                    }
                )
            else:
                return [i for i in range(int(max_int + 1))]

        # Initiate constraints for battery swap, pick-up and drop-off
        pick_ups = min(
            max(
                len(vehicle.current_location.scooters)
                - vehicle.current_location.ideal_state,
                0,
            ),
            vehicle.scooter_inventory_capacity - len(vehicle.scooter_inventory),
            vehicle.battery_inventory,
        )
        swaps = vehicle.get_max_number_of_swaps()
        drop_offs = max(
            min(
                vehicle.current_location.ideal_state
                - len(vehicle.current_location.scooters),
                len(vehicle.scooter_inventory),
            ),
            0,
        )
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
            if scooter.battery >= 70
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
                    swappable_scooters_id[:int(battery_swap)],
                    choose_pick_up(battery_swap, pick_up),
                    [scooter.id for scooter in vehicle.scooter_inventory][
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
        for cluster in state.stations:
            current_states.append(get_current_state(cluster))
            available_scooters.append(cluster.get_available_scooters())
        return current_states, available_scooters

    def get_best_action(self, simul, vehicle):
        # Find all possible actions
        actions = get_possible_actions(
            simul.state,
            vehicle,
            divide=self.get_possible_actions_divide,
            time=simul.time,
            number_of_neighbours=self.number_of_neighbors,
        )
        state = simul.state
        cache = EpsilonGreedyValueFunctionPolicy.get_cache(state)
        # Get state representation of current state
        state_features = self.value_function.get_state_features(
            simul.state, vehicle, cache
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
                forward_state: sim.State = copy.deepcopy(state)
                forward_vehicle: sim.Vehicle = forward_state.get_vehicle_by_id(
                    vehicle.id
                )
                # perform action
                forward_state.do_action(action, forward_vehicle, simul.time)
                # Simulate the system to generate potential lost trips
                _, _, lost_demands = policies.epsilon_greedy_value_function_policy.system_simulation.scripts.system_simulate(
                    forward_state
                )
                # Record lost trip rewards
                reward = (
                    sum(map(lambda lost_trips: lost_trips[0], lost_demands))
                    if len(lost_demands) > 0
                    else 0
                )
                # Find all actions after taking the action moving the state to s_{t+1}
                next_action_actions = get_possible_actions(forward_state,
                    forward_vehicle,
                    divide=self.get_possible_actions_divide,
                    exclude=[action.next_location],
                    time=simul.time
                    + action.get_action_time(
                        state.get_distance(
                            vehicle.current_location.id,
                            forward_vehicle.current_location.id,
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
