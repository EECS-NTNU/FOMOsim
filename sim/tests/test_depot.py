import unittest
import sim
import clustering.scripts
import decision
from globals import (
    VAN_BATTERY_INVENTORY,
    SMALL_DEPOT_CAPACITY,
    SWAP_TIME_PER_BATTERY,
    CHARGE_TIME_PER_BATTERY,
)


class DepotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = sim.World(
            shift_duration=VAN_BATTERY_INVENTORY * SWAP_TIME_PER_BATTERY + 1,
            policy=decision.SwapAllPolicy(),
            initial_state=clustering.scripts.get_initial_state(
                sample_size=100, number_of_clusters=10, initial_location_depot=True
            ),
            verbose=False,
            visualize=False,
        )
        self.vehicle = self.world.state.vehicles[0]

    def test_depot_charge(self):
        depot = sim.Depot(lat=0.0, lon=0.0, depot_id=0)
        depot.swap_battery_inventory(0, number_of_battery_to_change=10)
        self.assertLess(depot.get_available_battery_swaps(time=1), SMALL_DEPOT_CAPACITY)
        self.assertEqual(
            depot.get_available_battery_swaps(time=CHARGE_TIME_PER_BATTERY + 1),
            SMALL_DEPOT_CAPACITY,
        )

    def test_vehicle_battery_inventory_change(self):
        self.vehicle.battery_inventory = 0
        self.world.run()
        self.assertGreater(self.vehicle.battery_inventory, 0)


if __name__ == "__main__":
    unittest.main()
