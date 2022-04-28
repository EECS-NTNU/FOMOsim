import unittest

from classes import Depot, Station
from clustering.scripts import get_initial_state


class ClusteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state_mid = get_initial_state(500)
        self.state_small = get_initial_state(100)
        self.state_big = get_initial_state(2000, initial_location_depot=False)
        self.all_states = [self.state_small, self.state_mid, self.state_big]

    def test_sample_size(self):
        self.assertEqual(
            len(get_initial_state(200, 10, cache=False, save=False).get_scooters()),
            200,
        )

    def test_current_cluster_is_depot(self):
        for vehicle in get_initial_state(100).vehicles:
            self.assertIsInstance(vehicle.current_location, Depot)

    def test_current_cluster_is_not_depot(self):
        for vehicle in self.state_big.vehicles:
            self.assertIsInstance(vehicle.current_location, Station)

    def test_create_multiple_vehicles(self):
        multiple_vehicle_state = get_initial_state(
            100, number_of_vans=2,
        )
        # There is a total of 5 vehicles
        self.assertEqual(len(multiple_vehicle_state.vehicles), 5)
        # No vehicle has the same id
        for vehicle in multiple_vehicle_state.vehicles:
            self.assertEqual(
                1,
                len(
                    [
                        id_vehicle
                        for id_vehicle in multiple_vehicle_state.vehicles
                        if id_vehicle.id == vehicle.id
                    ]
                ),
            )
        # There are three vehicles with no scooter capacity
        self.assertEqual(
            3,
            len(
                [
                    vehicle
                    for vehicle in multiple_vehicle_state.vehicles
                    if vehicle.scooter_inventory_capacity == 0
                ]
            ),
        )
        # There are two vehicles with scooter capacity
        self.assertEqual(
            2,
            len(
                [
                    vehicle
                    for vehicle in multiple_vehicle_state.vehicles
                    if vehicle.scooter_inventory_capacity > 0
                ]
            ),
        )

    def test_move_probabilities(self):
        for state in [self.state_mid, self.state_small, self.state_big]:
            for cluster in state.stations:
                self.assertAlmostEqual(sum(cluster.move_probabilities), 1)
                self.assertFalse(
                    any([prob < 0 for prob in cluster.move_probabilities]),
                    "There are negative probabilities in the move probabilities matrix",
                )
                self.assertFalse(
                    any([prob > 1 for prob in cluster.move_probabilities]),
                    "There are probabilities bigger than one in the move probabilities matrix",
                )

if __name__ == "__main__":
    unittest.main()
