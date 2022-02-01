"""
A file for testing stuff without dealing with circular imports
"""
from clustering.scripts import get_initial_state
from system_simulation.scripts import system_simulate
from visualization.visualizer import *
from analysis.evaluate_policies import run_analysis
import unittest
import decision


class BasicVisualizerTests(unittest.TestCase):
    @staticmethod
    def test_state_and_flow_between_clusters():
        # state and flow between clusters visualization
        world = classes.World(
            5, None, get_initial_state(sample_size=100, number_of_clusters=6)
        )

        flows, trips, _ = system_simulate(world.state)

        visualize_cluster_flow(world.state, flows)

    @staticmethod
    def test_scooter_trips():
        # scooter trips visualization
        world = classes.World(
            5, None, get_initial_state(sample_size=20, number_of_clusters=5)
        )

        next_world = copy.deepcopy(world)

        flows, scooter_trips, _ = next_world.system_simulate()

        world.state.visualize_system_simulation(scooter_trips)

    @staticmethod
    def test_visualize_clusters():
        current_state = get_initial_state(sample_size=1000, number_of_clusters=20)
        current_state.visualize_clustering()

    @staticmethod
    def test_analysis():
        # test the analysis plot
        run_analysis(
            [
                classes.World(
                    21,
                    decision.SwapAllPolicy(),
                    get_initial_state(sample_size=100, number_of_clusters=10),
                )
            ],
            runs_per_policy=1,
        )


if __name__ == "__main__":
    unittest.main()
