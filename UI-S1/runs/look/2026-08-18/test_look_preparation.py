import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import look_common as common


class LookPreparationTest(unittest.TestCase):
    def test_geometry_contains_centroids_and_aspect(self):
        result = common.confrontation_window(3840, 2160, [[1000, 1000], [1500, 1100]])
        self.assertEqual(result["status"], "FEASIBLE")
        self.assertTrue(all(result["centroids_contained"]))
        width, height = result["dimensions"]
        self.assertLess(abs(width / height - common.ASPECT_RATIO), 0.01)

    def test_geometry_infeasible(self):
        result = common.confrontation_window(100, 100, [[0, 0], [99, 99]])
        self.assertEqual(result["status"], "INFEASIBLE_TOO_LARGE")

    def test_null_seed_deterministic(self):
        self.assertEqual(common.null_seed("row", 3), common.null_seed("row", 3))
        self.assertNotEqual(common.null_seed("row", 3), common.null_seed("row", 4))

    def test_allocation(self):
        allocation = common.allocate_counts({"a": 10, "b": 20}, 15)
        self.assertEqual(sum(allocation.values()), 15)
        self.assertTrue(all(value >= 1 for value in allocation.values()))

    def test_nearest_tie_prefers_first(self):
        self.assertEqual(common.nearest_choice([1, 0], [[0, 0], [2, 0]]), 0)
        self.assertIsNone(common.nearest_choice(None, [[0, 0], [2, 0]]))


if __name__ == "__main__":
    unittest.main()