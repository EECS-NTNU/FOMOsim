import random
import unittest

import classes
from clustering.scripts import get_initial_state
import globals


class StateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state_mid = get_initial_state(500)
        self.state_small = get_initial_state(100)
        self.state_big = get_initial_state(2000)

    def test_leave_probabilities(self):
        for state in [self.state_mid, self.state_small, self.state_mid]:
            for cluster in state.clusters:
                self.assertAlmostEqual(sum(cluster.get_leave_distribution()), 1)
                self.assertFalse(
                    any([prob < 0 for prob in cluster.get_leave_distribution()]),
                    "There are negative probabilities in the move probabilities matrix",
                )
                self.assertFalse(
                    any([prob > 1 for prob in cluster.get_leave_distribution()]),
                    "There are probabilities bigger than one in the move probabilities matrix",
                )
                self.assertEqual(0, cluster.get_leave_distribution()[cluster.id])


if __name__ == "__main__":
    unittest.main()
