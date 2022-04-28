from .abstract import *
from policies.haflan_haga_spetalen.value_functions.ANN import ANN


class ANNValueFunction(ValueFunction):
    """
    Class for the Artificial Neural Network function approximation. This class use the ANN class to call keras model
    methods.
    """

    def __init__(
        self,
        learning_rate,
        weight_init_value,
        discount_factor,
        vehicle_inventory_step_size,
        location_repetition,
        trace_decay,
        network_structure: [int],
    ):
        super().__init__(
            learning_rate,
            weight_init_value,
            discount_factor,
            vehicle_inventory_step_size,
            location_repetition,
            trace_decay,
        )
        self.network_structure = network_structure
        self.model = None
        self.train_count = 0

    def setup(self, state: sim.State):
        if self.setup_complete:
            return
        number_of_state_features = (
            self.get_number_of_location_indicators_and_state_features(state)
        )
        self.model = ANN(
            self.network_structure,
            number_of_state_features,
            self.trace_decay,
            self.discount_factor,
            self.step_size,
        )
        super(ANNValueFunction, self).setup(state)

    def use_replay_buffer(self):
        return True

    def train(self, rng, training_input):
        buffer_size = training_input
        if (
            len(self.replay_buffer) < buffer_size
            or len(self.replay_buffer_negative) < buffer_size
        ):
            return
        buffer_size = min(buffer_size, 64)
        for j in range(2):
            random_sample = (
                rng.choice(self.replay_buffer, buffer_size, replace=False)
                if j == 0
                else rng.choice(self.replay_buffer_negative, buffer_size, replace=False)
            )
            # Create training data from random sample
            states, targets = [], []
            for i, (
                state_features,
                reward,
                next_state_features,
            ) in enumerate(random_sample):
                states.append(state_features)
                next_state_value = self.model.predict(next_state_features)
                targets.append(self.discount_factor * next_state_value + reward)
            self.model.batch_fit(states, targets, verbose=1, batch_size=buffer_size)

        if self.train_count % 1:
            self.model.update_predict_model()
        self.train_count += 1

    def estimate_value(
        self,
        state: sim.State,
        day, hour, 
        vehicle: sim.Vehicle,
        time: int,
    ):

        return self.estimate_value_from_state_features(
            self.get_state_features(state, day, hour, vehicle, time)
        )

    def estimate_value_from_state_features(self, state_features: [float]):
        return self.model.predict(state_features)

    def update_weights(
        self,
        current_state_features: [float],
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):
        td_error = self.compute_and_record_td_error(
            current_state_value, next_state_value, reward
        )
        self.model.fit(
            [current_state_features],
            [td_error + current_state_value],
            epochs=10,
        )

    def get_next_state_features(
        self,
        state: sim.State,
        day, hour, 
        vehicle: sim.Vehicle,
        action: sim.Action,
        cache=None,  # current_states, available_scooters = cache
    ):
        return self.convert_next_state_features(state, day, hour, vehicle, action, cache)

    def get_state_features(
        self, state: sim.State, day, hour, vehicle: sim.Vehicle, cache=None
    ):
        return self.convert_state_to_features(state, day, hour, vehicle, cache=cache)

    def __str__(self):
        return f"ANNValueFunction - {self.shifts_trained}"
