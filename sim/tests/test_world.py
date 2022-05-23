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

    def save_world(self):
        filepath = f"{settings.WORLD_CACHE_DIR}/{self.world.get_filename()}.pickle"
        self.world.save(settings.WORLD_CACHE_DIR)
        file_world = sim.World.load(filepath)
        file_world.duration = 2
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
