import os
import unittest
import random
import sim
import clustering.scripts
import decision
import decision.value_functions
import settings


class WorldTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.world = sim.World(
            40,
            policy=decision.SwapAllPolicy(),
            initial_state=clustering.scripts.get_initial_state(100, 20),
            visualize=False,
        )

    def test_run(self):
        self.world.event_queue = [sim.Event(time) for time in range(10, 41, 10)]
        self.world.run()

    def test_add_event(self):
        # Clear initial stack
        self.world.event_queue = []
        shuffled_events = [sim.Event(time) for time in range(10, 41, 10)]
        random.shuffle(shuffled_events)
        for event in shuffled_events:
            self.world.add_event(event)
        self.assertSequenceEqual(
            [event.time for event in self.world.event_queue], range(10, 41, 10)
        )

    def test_run_with_initial_stack(self):
        self.world.run()

    def test_tabu_list(self):
        # Clear initial stack
        self.world.event_queue = []
        # Perform Vehicle arrival event
        arrival_event = sim.VehicleArrival(0, 0, False)
        arrival_event.perform(self.world)
        vehicle = [vehicle for vehicle in self.world.state.vehicles if vehicle.id == 0][
            0
        ]
        # Record the next location
        first_vehicle_location = vehicle.current_location.id

        # Check that next vehicle location is in tabu list
        self.assertIn(first_vehicle_location, self.world.tabu_list)
        # Check that exclude works in next vehicle location not in possible actions
        self.assertNotIn(
            first_vehicle_location,
            [
                action.next_location
                for action in self.world.state.get_possible_actions(
                    vehicle, number_of_neighbours=10, exclude=self.world.tabu_list
                )
            ],
        )
        # Perform vehicle next vehicle arrival event
        self.world.event_queue.pop().perform(self.world)
        # Check that the old vehicle location is not in the tabu list
        self.assertNotIn(first_vehicle_location, self.world.tabu_list)

    def save_world(self):
        filepath = f"{settings.WORLD_CACHE_DIR}/{self.world.get_filename()}.pickle"
        self.world.save(settings.WORLD_CACHE_DIR)
        file_world = sim.World.load(filepath)
        file_world.shift_duration = 2
        file_world.run()
        os.remove(filepath)
        return file_world

    def test_save_world_linear(self):
        # Change weights in value function
        self.world.policy = self.world.set_policy(
            policy_class=decision.EpsilonGreedyValueFunctionPolicy,
            value_function_class=decision.value_functions.LinearValueFunction,
        )
        self.save_world()

    def test_save_world_ann(self):
        # Change weights in value function
        self.world.policy = self.world.set_policy(
            policy_class=decision.EpsilonGreedyValueFunctionPolicy,
            value_function_class=decision.value_functions.ANNValueFunction,
        )
        self.save_world()


if __name__ == "__main__":
    unittest.main()
