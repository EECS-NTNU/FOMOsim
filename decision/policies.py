"""
This file contains all the policies used in the thesis.
"""
import copy

import classes
import numpy.random as random
import abc

import system_simulation.scripts


class Policy(abc.ABC):
    """
    Base Policy class. Used mainly as an interface for the get best action method
    """

    def __init__(
        self,
        get_possible_actions_divide,
        number_of_neighbors,
    ):
        self.get_possible_actions_divide = get_possible_actions_divide
        self.number_of_neighbors = number_of_neighbors

    @abc.abstractmethod
    def get_best_action(self, world, vehicle):
        """
        Returns the best action for the input vehicle in the world context
        :param world: world object that contains the whole world state
        :param vehicle: the vehicle to perform an action
        :return: the best action according to the policy
        """
        pass

    def setup_from_state(self, state):
        """
        Function to be called after association with a state object is created.
        Nice place to setup value functions.
        :param state: state object associated with policy
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @staticmethod
    def print_action_stats(
        world,
        vehicle: classes.Vehicle,
        actions_info: [(classes.Action, int, int)],
    ) -> None:
        if world.verbose:
            print(f"\n{vehicle}:")
            for action, reward, computational_time in actions_info:
                print(
                    f"\n{action} Reward - {round(reward, 3)} | Comp. time - {round(computational_time, 2)}"
                )
            print("\n----------------------------------------------------------------")


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
        super().__init__(get_possible_actions_divide, number_of_neighbors)
        self.value_function = value_function
        self.epsilon = epsilon
        self.decision_times = []

    @staticmethod
    def get_cache(state):
        # Cache current states in state
        current_states, available_scooters = [], []
        for cluster in state.clusters:
            current_states.append(cluster.get_current_state())
            available_scooters.append(cluster.get_available_scooters())
        return current_states, available_scooters

    def get_best_action(self, world, vehicle):
        # Find all possible actions
        actions = world.state.get_possible_actions(
            vehicle,
            divide=self.get_possible_actions_divide,
            exclude=world.tabu_list,
            time=world.time,
            number_of_neighbours=self.number_of_neighbors,
        )
        state = world.state
        cache = EpsilonGreedyValueFunctionPolicy.get_cache(state)
        # Get state representation of current state
        state_features = self.value_function.get_state_features(
            world.state, vehicle, cache
        )

        # Epsilon greedy choose an action based on value function
        if self.epsilon > random.rand():
            best_action = random.choice(actions)
        else:
            # Create list containing all actions and their rewards and values (action, reward, value_function_value)
            action_info = [
                (
                    classes.Action([], [], [], random.choice(world.state.locations).id),
                    -1000,
                    [],
                )  # No actions bug
            ]
            reward = 0
            for action in actions:
                # look one action ahead
                forward_state: classes.State = copy.deepcopy(state)
                forward_vehicle: classes.Vehicle = forward_state.get_vehicle_by_id(
                    vehicle.id
                )
                # perform action
                forward_state.do_action(action, forward_vehicle, world.time)
                # Simulate the system to generate potential lost trips
                _, _, lost_demands = system_simulation.scripts.system_simulate(
                    forward_state
                )
                # Record lost trip rewards
                reward = (
                    sum(map(lambda lost_trips: lost_trips[0], lost_demands))
                    if len(lost_demands) > 0
                    else 0
                )
                # Find all actions after taking the action moving the state to s_{t+1}
                next_action_actions = forward_state.get_possible_actions(
                    forward_vehicle,
                    divide=self.get_possible_actions_divide,
                    exclude=world.tabu_list + [action.next_location],
                    time=world.time
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
                        next_state_value + reward * world.LOST_TRIP_REWARD,
                        next_state_features,
                    )
                )
            # Choose the action with the highest value and reward
            best_action, next_state_value, next_state_features = max(
                action_info, key=lambda pair: pair[1]
            )
            if not world.disable_training:
                if self.value_function.use_replay_buffer():
                    self.value_function.train(world.REPLAY_BUFFER_SIZE)
                else:
                    self.value_function.train(
                        (
                            state_features,
                            reward * world.LOST_TRIP_REWARD,
                            next_state_features,
                        )
                    )
        return best_action, state_features

    def setup_from_state(self, state):
        self.value_function.setup(state)

    def __str__(self):
        return f"EpsilonGreedyPolicy w/ {self.value_function.__str__()}"


"""
SECONDARY POLICIES
"""


class SwapAllPolicy(Policy):
    def __init__(self):

        super().__init__(1, 1)

    def get_best_action(self, world, vehicle):
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
            next_location = world.state.depots[0]
        else:
            next_location = sorted(
                [
                    cluster
                    for cluster in world.state.clusters
                    if cluster.id != vehicle.current_location.id
                    and cluster.id not in world.tabu_list
                ],
                key=lambda cluster: (
                    len(cluster.scooters) - cluster.get_current_state()
                )
                / (len(cluster.scooters) + 1),
                reverse=True,
            )[0]

        # Return an action with no re-balancing, only scooter swapping
        return classes.Action(
            battery_swaps=swappable_scooters_ids[:number_of_scooters_to_swap],
            pick_ups=[],
            delivery_scooters=[],
            next_location=next_location.id,
        )


class RandomActionPolicy(Policy):
    def __init__(self, get_possible_actions_divide, number_of_neighbors):
        super().__init__(get_possible_actions_divide, number_of_neighbors)

    def get_best_action(self, world, vehicle):
        # all possible actions in this state
        possible_actions = world.state.get_possible_actions(
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
          return classes.Action([], [], [], 0)


class DoNothing(Policy):
    def __init__(self):
        super().__init__(0, 0)

    def get_best_action(self, world, vehicle):
        return classes.Action([], [], [], 0)


class NightShift(Policy):
    def __init__(self):
        super().__init__(0, 0)

    def get_best_action(self, world, vehicle):
        return classes.Action([], [], [], 0)


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
