import copy
import unittest
import random

import numpy as np

import classes
import clustering.scripts
import decision
import decision.value_functions
import globals
import system_simulation.scripts
from classes import World, Action, Scooter
from clustering.scripts import get_initial_state
from decision.neighbour_filtering import filtering_neighbours


class BasicDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_state = get_initial_state(
            sample_size=100, number_of_clusters=2, initial_location_depot=False
        )
        self.vehicle = self.initial_state.vehicles[0]
        self.number_of_neighbours = 4

    def test_battery_swaps(self):
        # Modify initial state. 5 battery swaps possible.
        self.vehicle.scooter_inventory = []
        # Let the current location of the vehicle contain 5 scooters
        self.vehicle.current_location.scooters = self.vehicle.current_location.scooters[
            :5
        ]
        self.vehicle.current_location.ideal_state = 5
        start_number_of_scooters = len(self.vehicle.current_location.scooters)
        current_cluster = self.vehicle.current_location

        for scooter in self.vehicle.current_location.scooters:
            scooter.battery = 30.0
        start_battery_percentage = current_cluster.get_current_state() * 100

        # Get all possible actions
        actions = self.initial_state.get_possible_actions(
            self.vehicle,
            number_of_neighbours=6,
        )

        # Test number of swaps less or equal to ideal state
        for action in actions:
            self.assertLessEqual(
                len(action.battery_swaps),
                self.vehicle.current_location.ideal_state,
            )

        # Test number of actions
        self.assertEqual(len(actions), 5)

        self.initial_state.do_action(actions[-1], self.vehicle, 0)

        # Test number of scooters
        self.assertEqual(len(current_cluster.scooters), start_number_of_scooters)

        # Test battery percentage
        self.assertEqual(
            current_cluster.get_current_state() * 100,
            start_battery_percentage + len(actions[-1].battery_swaps) * 70.0,
        )

    def test_pick_ups(self):
        # Modify initial state. 5 battery swaps and 2 pick ups possible
        self.vehicle.scooter_inventory = []
        self.vehicle.current_location.scooters = self.vehicle.current_location.scooters[
            :5
        ]
        self.vehicle.current_location.ideal_state = 3
        start_number_of_scooters = len(self.vehicle.current_location.scooters)
        current_cluster = self.vehicle.current_location

        for scooter in self.vehicle.current_location.scooters:
            scooter.battery = 20.0
        start_battery_percentage = current_cluster.get_current_state() * 100

        # Get all possible actions
        actions = self.initial_state.get_possible_actions(
            self.vehicle, number_of_neighbours=1
        )

        # Test number of actions
        self.assertEqual(len(actions), 14)

        # Test no reward for pickup
        self.initial_state.do_action(actions[-1], self.vehicle, 0)

        # Test number of scooters
        self.assertEqual(
            len(current_cluster.scooters),
            start_number_of_scooters - len(actions[-1].pick_ups),
        )

        # Test inventory vehicle
        self.assertEqual(
            start_number_of_scooters - len(current_cluster.scooters),
            len(self.vehicle.scooter_inventory),
        )

        # Test battery percentage
        self.assertAlmostEqual(
            current_cluster.get_current_state() * 100,
            start_battery_percentage
            + len(actions[-1].battery_swaps) * 80.0
            - len(actions[-1].pick_ups) * 20.0,
        )

    def test_deliveries(self):
        # Modify initial state. 5 battery swaps and 2 drop-offs possible
        self.vehicle.scooter_inventory = self.vehicle.current_location.scooters[7:9]
        self.vehicle.current_location.scooters = self.vehicle.current_location.scooters[
            :5
        ]
        self.vehicle.current_location.ideal_state = 7
        start_number_of_scooters = len(self.vehicle.current_location.scooters)
        current_cluster = self.vehicle.current_location

        for scooter in self.vehicle.current_location.scooters:
            scooter.battery = 30.0
        start_battery_percentage = current_cluster.get_current_state() * 100

        # Get all possible actions
        actions = self.initial_state.get_possible_actions(
            self.vehicle, number_of_neighbours=6
        )

        # Test number of actions
        self.assertEqual(len(actions), 17)

        self.initial_state.do_action(actions[-1], self.vehicle, 0)

        # Test number of scooters
        self.assertEqual(
            len(current_cluster.scooters),
            start_number_of_scooters + len(actions[-1].delivery_scooters),
        )

        # Test battery percentage
        delivery_scooter_objects = [
            scooter
            for scooter in current_cluster.scooters
            if scooter.id in actions[-1].delivery_scooters
        ]
        delivery_scooter_battery = sum(
            map(
                lambda delivery_scooter: delivery_scooter.battery,
                delivery_scooter_objects,
            )
        )
        self.assertAlmostEqual(
            current_cluster.get_current_state() * 100,
            start_battery_percentage
            + len(actions[-1].battery_swaps) * 70.0
            + delivery_scooter_battery,
        )

    def test_number_of_actions_clusters(self):
        initial_state = get_initial_state(
            sample_size=100, number_of_clusters=6, initial_location_depot=False
        )
        vehicle = initial_state.vehicles[0]
        # Modify initial state. 1 battery swap and 0 drop-offs possible
        vehicle.scooter_inventory_capacity = 0
        vehicle.current_location.scooters = vehicle.current_location.scooters[:1]

        # Get all possible actions
        actions = initial_state.get_possible_actions(
            vehicle,
            number_of_neighbours=2,
        )

        # Test number of actions possible
        self.assertEqual(2, len(actions))

    def test_number_of_actions(self):
        bigger_state = get_initial_state(sample_size=1000, initial_location_depot=False)
        vehicle = bigger_state.vehicles[0]
        vehicle.current_location = random.choice(
            [
                cluster
                for cluster in bigger_state.clusters
                if cluster.number_of_scooters() > 0
            ]
        )
        self.assertLess(
            len(
                bigger_state.get_possible_actions(
                    vehicle, self.number_of_neighbours, divide=2
                )
            ),
            len(bigger_state.get_possible_actions(vehicle, self.number_of_neighbours)),
        )
        self.assertLess(
            len(
                bigger_state.get_possible_actions(
                    vehicle, self.number_of_neighbours, divide=2
                )
            ),
            len(
                bigger_state.get_possible_actions(
                    vehicle, self.number_of_neighbours, divide=4
                )
            ),
        )


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(40, None, clustering.scripts.get_initial_state(100, 10))

    def test_swap_all_policy(self):
        self.world.policy = decision.SwapAllPolicy()
        vehicle_swap_all_policy = self.world.state.vehicles[0]
        action = self.world.policy.get_best_action(self.world, vehicle_swap_all_policy)
        self.assertIsInstance(action, Action)
        self.assertEqual(len(action.pick_ups), 0)
        self.assertEqual(len(action.delivery_scooters), 0)


# helper function to update the value function (call two times in ann test)
def update_value_function(value_function, state_features, next_state_features, reward):
    state_value = value_function.estimate_value_from_state_features(state_features)
    next_state_value = value_function.estimate_value_from_state_features(
        next_state_features
    )
    value_function.update_weights(
        current_state_value=state_value,
        current_state_features=state_features,
        next_state_value=next_state_value,
        reward=reward,
    )


class ValueFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        hyper_params = globals.HyperParameters()
        self.value_function_args = (
            hyper_params.WEIGHT_UPDATE_STEP_SIZE,
            hyper_params.WEIGHT_INITIALIZATION_VALUE,
            hyper_params.DISCOUNT_RATE,
            hyper_params.VEHICLE_INVENTORY_STEP_SIZE,
            hyper_params.LOCATION_REPETITION,
            hyper_params.TRACE_DECAY,
        )
        self.world = World(
            100,
            initial_state=clustering.scripts.get_initial_state(
                1000, 30, initial_location_depot=False
            ),
            policy=None,
        )

    def world_value_function_check(self, value_function):
        # No discount should give reward equal to TD-error
        value_function.setup(self.world.state)
        vehicle = self.world.state.vehicles[0]
        action = decision.policies.SwapAllPolicy().get_best_action(self.world, vehicle)
        state = copy.deepcopy(self.world.state)
        state_features = value_function.get_state_features(state, vehicle)
        copied_vehicle = copy.deepcopy(vehicle)
        reward = action.get_reward(
            vehicle,
            self.world.LOST_TRIP_REWARD,
            self.world.DEPOT_REWARD,
            self.world.VEHICLE_INVENTORY_STEP_SIZE,
            self.world.PICK_UP_REWARD,
        )
        self.world.state.do_action(action, vehicle, self.world.time)
        for i in range(100):
            state_value = value_function.estimate_value(state, copied_vehicle, 0)
            next_state_value = value_function.estimate_value(
                self.world.state, vehicle, self.world.time
            )
            value_function.update_weights(
                current_state_value=state_value,
                current_state_features=state_features,
                next_state_value=next_state_value,
                reward=reward,
            )
        # Check that the fist td errors are bigger than the last
        self.assertLess(
            abs(sum(value_function.td_errors[-3:]) / 3),
            abs(sum(value_function.td_errors[:3]) / 3),
        )

    def ann_learning(self, value_function):
        value_function.setup(self.world.state)
        self.world.LOST_TRIP_REWARD = -1

        # Creating a list of states with associated negative reward
        simulation_state = copy.deepcopy(self.world.state)
        vehicle = simulation_state.vehicles[0]
        system_simulated_states = []
        i = 0
        # simulating to provoke lost demand
        while len(system_simulated_states) < 10:
            _, _, lost_demand = system_simulation.scripts.system_simulate(
                simulation_state
            )
            # recording state and lost reward if there was lost demand after simulation
            if len(lost_demand) > 0:
                system_simulated_states.append(
                    (
                        value_function.get_state_features(
                            simulation_state,
                            vehicle,
                            i * globals.ITERATION_LENGTH_MINUTES,
                        ),
                        sum([lost_demand for lost_demand, _ in lost_demand])
                        * self.world.LOST_TRIP_REWARD,
                    )
                )

            i += 1

        # simulating doing actions that yields positive reward
        # (swap battery in clusters with available scooters less than ideal state)
        unsimulated_world = copy.deepcopy(self.world)
        accumulated_action_time = 0
        unsimulated_states = []
        # recording clusters with available scooters less than ideal state
        deficient_cluster = [
            cluster
            for cluster in unsimulated_world.state.clusters
            if len(cluster.get_available_scooters()) < cluster.ideal_state
        ]
        counter = 0
        vehicle = unsimulated_world.state.vehicles[0]
        # safety break if internal break doesn't apply
        while counter < len(deficient_cluster) and len(unsimulated_states) < 10:
            # swapping batteries on the n-th cluster in deficient cluster list
            cluster = deficient_cluster[counter]
            vehicle.battery_inventory = vehicle.battery_inventory_capacity
            vehicle.current_location = cluster
            # creating an action to swap all batteries and recording the state and reward
            action = classes.Action(
                [scooter.id for scooter in cluster.get_swappable_scooters()][
                    : vehicle.battery_inventory
                ],
                [],
                [],
                deficient_cluster[counter + 1].id,
            )
            reward = action.get_reward(
                vehicle,
                0,
                self.world.DEPOT_REWARD,
                self.world.VEHICLE_INVENTORY_STEP_SIZE,
                self.world.PICK_UP_REWARD,
            )
            unsimulated_states.append(
                (
                    value_function.get_state_features(
                        unsimulated_world.state, vehicle, accumulated_action_time
                    ),
                    reward,
                )
            )
            # calculating action distance and action time so it can be used when getting state features
            # (unnecessary, but have to use a time when creating state features)
            action_distance = unsimulated_world.state.get_distance(
                vehicle.current_location.id, action.next_location
            )
            accumulated_action_time += unsimulated_world.state.do_action(
                action, vehicle, accumulated_action_time
            ) + action.get_action_time(action_distance)

            counter += 1

        # training two times on the positive and negative rewarded states
        for _ in range(2):
            for i in range(len(system_simulated_states) - 1):
                state_features, reward = system_simulated_states[i]
                next_state_features = system_simulated_states[i + 1][0]
                update_value_function(
                    value_function, state_features, next_state_features, reward
                )

            for i in range(len(unsimulated_states) - 1):
                state_features, reward = unsimulated_states[i]
                next_state_features = unsimulated_states[i + 1][0]
                update_value_function(
                    value_function, state_features, next_state_features, reward
                )

        # check if the ann predicts higher value for the positively rewarded state then the negative one
        self.assertGreater(
            value_function.estimate_value_from_state_features(unsimulated_states[0][0]),
            value_function.estimate_value_from_state_features(
                system_simulated_states[0][0]
            ),
        )

    def test_linear_value_function(self):
        self.world_value_function_check(
            decision.value_functions.LinearValueFunction(*self.value_function_args)
        )

    @unittest.skip  # skipping this as we now do experience replay
    def test_ann_value_function(self):
        self.ann_learning(
            decision.value_functions.ANNValueFunction(
                *self.value_function_args,
                [1000, 1000, 1000, 1000, 500, 100],
            )
        )

    def test_next_state_from_action(self):
        value_function = decision.value_functions.LinearValueFunction(
            *self.value_function_args
        )
        value_function.setup(self.world.state)
        # Record current state
        vehicle = self.world.state.vehicles[0]
        # Scooters is current cluster
        scooters = self.world.state.clusters[
            vehicle.current_location.id
        ].get_swappable_scooters()
        deliver_scooter = random.choice(
            [
                scooter
                for scooter in random.choice(
                    [
                        cluster
                        for cluster in self.world.state.clusters
                        if cluster.id != vehicle.current_location.id
                    ]
                ).scooters
            ]
        )
        vehicle.pick_up(deliver_scooter)
        # Action that does a bit of everything
        action = classes.Action(
            [scooter.id for scooter in scooters[:3]],
            [scooter.id for scooter in scooters[3:10]],
            [deliver_scooter.id],
            random.choice(
                [
                    cluster.id
                    for cluster in self.world.state.clusters
                    if cluster.id != vehicle.current_location.id
                ]
            ),
        )
        function_next_state_features = value_function.convert_next_state_features(
            self.world.state, vehicle, action
        )
        self.world.state.do_action(action, vehicle, self.world.time)
        next_state_features = value_function.convert_state_to_features(
            self.world.state, vehicle
        )
        self.assertEqual(len(function_next_state_features), len(next_state_features))
        for i, value in enumerate(function_next_state_features):
            self.assertAlmostEqual(
                function_next_state_features[i],
                next_state_features[i],
                msg=f"not equal at {i}",
            )


if __name__ == "__main__":
    unittest.main()
