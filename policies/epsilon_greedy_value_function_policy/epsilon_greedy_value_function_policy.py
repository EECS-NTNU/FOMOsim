"""
This file contains all the policies used in the thesis.
"""
import copy

import classes
import numpy.random as random
import abc

import system_simulation.scripts

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
