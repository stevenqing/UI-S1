import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("stage0", RUN_DIR / "stage0.py")
stage0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage0
SPEC.loader.exec_module(stage0)


class Stage0Test(unittest.TestCase):
    def test_rectangle_iou(self):
        self.assertEqual(stage0.rectangle_iou([0, 0, 2, 2], [0, 0, 2, 2]), 1.0)
        self.assertEqual(stage0.rectangle_iou([0, 0, 1, 1], [1, 1, 2, 2]), 0.0)

    def test_greedy_drop_tie_earliest(self):
        regions = [[0, 0, 2, 2], [0, 0, 2, 2], [3, 3, 5, 5]]
        self.assertEqual(stage0.greedy_drop(regions, 1), [0])

    def test_best_window_exact_and_tie(self):
        uncovered = np.asarray([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=bool)
        window, gain = stage0.best_window(uncovered, 2, 2)
        self.assertEqual(window, [0, 0, 2, 2])
        self.assertEqual(gain, 3)

    def test_complementary_updates(self):
        windows, gains = stage0.complementary_windows(4, 2, [[0, 0, 2, 2]], 2)
        self.assertEqual(windows[0], [2, 0, 4, 2])
        self.assertEqual(gains[0], 4)
        self.assertEqual(gains[1], 0)

    def test_contains_center_half_open(self):
        self.assertTrue(stage0.contains_center([0, 0, 2, 2], [0, 0, 2, 2]))
        self.assertFalse(stage0.contains_center([0, 0, 1, 1], [1, 1, 3, 3]))


if __name__ == "__main__":
    unittest.main()