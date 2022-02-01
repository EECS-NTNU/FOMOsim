import os
import unittest
import analysis.evaluate_policies
import analysis.train_value_function
import analysis.multiprocessing_training
import analysis.export_metrics_to_xlsx
import classes
import clustering.scripts
import decision
import decision.value_functions
import globals


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = classes.World(
            80,
            None,
            clustering.scripts.get_initial_state(100, 10),
            visualize=False,
            verbose=False,
            NUMBER_OF_NEIGHBOURS=5,
            TRAINING_SHIFTS_BEFORE_SAVE=1,
            MODELS_TO_BE_SAVED=2,
        )

    def delete_dir(self, training_directory):
        for world_file_path in os.listdir(training_directory):
            file_path = os.path.join(training_directory, world_file_path)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                self.assertTrue(
                    False,
                    "file not found. Possible that train value function did not create necessary files",
                )

        if os.path.isdir(training_directory):
            os.rmdir(training_directory)
        else:
            self.assertTrue(
                False,
                "file not found. Possible that train value function did not create necessary files",
            )

    def test_train_value_function(self):
        initial_epsilon = self.world.INITIAL_EPSILON
        self.world.policy = self.world.set_policy(
            policy_class=decision.EpsilonGreedyValueFunctionPolicy,
            value_function_class=decision.value_functions.LinearValueFunction,
        )
        analysis.train_value_function.train_value_function(self.world)
        self.assertLess(self.world.policy.epsilon, initial_epsilon)

        # Remove created files
        training_directory = os.path.join(
            globals.WORLD_CACHE_DIR, self.world.get_train_directory()
        )
        self.delete_dir(training_directory)

    @staticmethod
    @unittest.skip
    def test_export_to_excel():
        # running test instances and exporting them to excel
        analysis.evaluate_policies.run_analysis_from_path(
            "world_cache/test_models",
            shift_duration=10,
            export_to_excel=True,
            runs_per_policy=1,
        )
        file_name = f"computational_study/Test.xlsx"
        # removing the test file that was created during the test
        os.remove(file_name)


if __name__ == "__main__":
    unittest.main()
