import unittest

from stage0 import dual_origin_gains, theta_grid
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

    def test_dual_origin_keeps_gate_outside_baseline(self):
        rows = {
            "a": {"target_bbox": [0, 0, 1, 1]},
            "b": {"target_bbox": [0, 0, 1, 1]},
        }
        baselines = {"majority": {"a": True, "b": False}, "dev_selection": {"a": False, "b": True}}
        ocr = {"a": {"point": None, "score": 0}, "b": {"point": None, "score": 0}}
        report = dual_origin_gains(["a", "b"], baselines, ocr, 1.0, rows)
        self.assertEqual(report, {"majority": 0.0, "dev_selection": 0.0})


if __name__ == "__main__":
    unittest.main()