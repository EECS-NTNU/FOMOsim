import unittest

import classes
from clustering.scripts import get_initial_state


class BasicSystemSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = classes.World(1, None, get_initial_state(2500, 50))

    def test_equal_number_of_scooters(self):
        number_of_scooters_before = sum(
            map(lambda cluster: len(cluster.scooters), self.world.state.clusters)
        )

        self.world.system_simulate()
        number_of_scooters_after = sum(
            map(lambda cluster: len(cluster.scooters), self.world.state.clusters)
        )
        self.assertEqual(number_of_scooters_before, number_of_scooters_after)

    def test_less_or_equal_total_battery(self):
        total_battery_of_scooters_before = sum(
            map(
                lambda cluster: sum([scooter.battery for scooter in cluster.scooters]),
                self.world.state.clusters,
            )
        )

        self.world.system_simulate()
        total_battery_of_scooters_after = sum(
            map(
                lambda cluster: sum([scooter.battery for scooter in cluster.scooters]),
                self.world.state.clusters,
            )
        )
        self.assertLessEqual(
            total_battery_of_scooters_after, total_battery_of_scooters_before
        )

    def test_flows_and_trips(self):
        initial_cluster_inventory = {}

        for cluster in self.world.state.clusters:
            initial_cluster_inventory[cluster.id] = len(cluster.scooters)

        flows, trips, _ = self.world.system_simulate()

        out_flow = {cluster.id: 0 for cluster in self.world.state.clusters}
        in_flow = {cluster.id: 0 for cluster in self.world.state.clusters}

        trips_from = {cluster.id: 0 for cluster in self.world.state.clusters}
        trips_to = {cluster.id: 0 for cluster in self.world.state.clusters}

        for start, end, scooter in trips:
            trips_from[start.id] += 1
            trips_to[end.id] += 1

        for cluster in self.world.state.clusters:
            out_flow[cluster.id] = sum(
                [flow for start, end, flow in flows if start == cluster.id]
            )
            in_flow[cluster.id] = sum(
                [flow for start, end, flow in flows if end == cluster.id]
            )

            # Test flow in/out equal to change in scooter inventory in a cluster
            self.assertEqual(
                initial_cluster_inventory[cluster.id] - len(cluster.scooters),
                out_flow[cluster.id] - in_flow[cluster.id],
            )

            # Test number of trips in/out equal to change in scooter inventory in a cluster
            self.assertEqual(
                initial_cluster_inventory[cluster.id] - len(cluster.scooters),
                trips_from[cluster.id] - trips_to[cluster.id],
            )

        # Test that total flow out equal to total flow in
        self.assertEqual(sum(out_flow.values()), sum(in_flow.values()))


if __name__ == "__main__":
    unittest.main()
