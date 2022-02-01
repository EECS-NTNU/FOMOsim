import unittest
import decision.value_functions
import clustering.scripts
from classes import World
from training_simulation.scripts import training_simulation


class EpsilonGreedyPolicyTrainingTest(unittest.TestCase):
    def training_simulation(self, starts_at_depot):
        world = World(
            80,
            None,
            clustering.scripts.get_initial_state(
                100, 20, initial_location_depot=starts_at_depot
            ),
            visualize=False,
            REPLAY_BUFFER_SIZE=1,
        )
        world.policy = world.set_policy(
            policy_class=decision.EpsilonGreedyValueFunctionPolicy,
            value_function_class=decision.value_functions.ANNValueFunction,
        )
        training_simulation(world)

    def test_start_in_depot(self):
        self.training_simulation(True)

    def test_start_in_cluster(self):
        self.training_simulation(False)


if __name__ == "__main__":
    unittest.main()
