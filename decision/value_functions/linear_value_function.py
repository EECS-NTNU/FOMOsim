import itertools

import numpy as np

from .abstract import *


class LinearValueFunction(ValueFunction):
    """
    Class for the value function approximation with the linear model
    """

    def __init__(
        self,
        weight_update_step_size,
        weight_init_value,
        discount_factor,
        vehicle_inventory_step_size,
        location_repetition,
        trace_decay,
    ):
        super().__init__(
            weight_update_step_size,
            weight_init_value,
            discount_factor,
            vehicle_inventory_step_size,
            location_repetition,
            trace_decay,
        )
        self.weights = []
        self.eligibilities = None

    def use_replay_buffer(self):
        return False

    def train(self, training_input):
        state_features, reward, next_state_features = training_input
        self.update_weights(
            state_features,
            self.estimate_value_from_state_features(state_features),
            self.estimate_value_from_state_features(next_state_features),
            reward,
        )

    def setup(self, state):
        if self.setup_complete:
            return
        number_of_state_features = (
            self.get_number_of_location_indicators_and_state_features(state)
        )
        self.weights = [self.weight_init_value] * (1 + number_of_state_features)  # Bias
        self.reset_eligibilities()
        super(LinearValueFunction, self).setup(state)

    def estimate_value(
        self,
        state,
        vehicle,
        time,
    ):
        return self.estimate_value_from_state_features(
            self.get_state_features(state, vehicle)
        )

    def estimate_value_from_state_features(self, state_features: [float]):
        return float(np.dot(self.weights, [1] + state_features))

    def update_weights(
        self,
        current_state_features: [float],
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):

        self.eligibilities = (
            self.discount_factor * self.trace_decay * self.eligibilities
            + ([1] + current_state_features)
        )

        self.weights += np.multiply(
            self.step_size
            * self.compute_and_record_td_error(
                current_state_value, next_state_value, reward
            ),
            self.eligibilities,
        )

    def reset_eligibilities(self):
        self.eligibilities = np.zeros(len(self.weights))

    @Decorators.check_setup
    def create_location_features_combination(self, state_features):

        locations_features_combination = list(
            [
                factor1 * factor2
                for factor1, factor2 in itertools.combinations(state_features, 2)
            ]
        )

        return [1] + state_features + locations_features_combination

    def get_state_features(self, state, vehicle, cache=None):
        return self.convert_state_to_features(state, vehicle, cache=cache)

    def get_next_state_features(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        action: classes.Action,
        cache=None,  # current_states, available_scooters = cache
    ):
        return self.convert_next_state_features(state, vehicle, action, cache)

    def __str__(self):
        return f"LinearValueFunction - {self.shifts_trained}"
