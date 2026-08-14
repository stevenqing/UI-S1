import unittest

from stage0 import theta_grid
from stage1 import grouped_bootstrap


class StageNumericsTest(unittest.TestCase):
    def test_theta_grid_has_twelve_ordered_values(self):
        values = theta_grid([0, 0.1, 0.2, 0.3, 0.4])
        self.assertEqual(len(values), 12)
        self.assertEqual(values[-1], float("inf"))
        self.assertEqual(values[:-1], sorted(values[:-1]))

    def test_grouped_bootstrap_point(self):
        rows = [
            {"fold": 0, "application": "a", "left": True, "right": False},
            {"fold": 0, "application": "b", "left": False, "right": False},
            {"fold": 1, "application": "c", "left": True, "right": True},
        ]
        report = grouped_bootstrap(rows, "left", "right", 7, resamples=100)
        self.assertAlmostEqual(report["point"], 1 / 3)


if __name__ == "__main__":
    unittest.main()