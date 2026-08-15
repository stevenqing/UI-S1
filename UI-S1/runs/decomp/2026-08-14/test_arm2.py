import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("arm2", RUN_DIR / "arm2.py")
arm2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arm2
SPEC.loader.exec_module(arm2)


def candidate(order, coordinate, action="POINT"):
    return {"order": order, "coordinate": coordinate, "action": action, "parameter": "", "parse_ok": True}


class Arm2Test(unittest.TestCase):
    def test_complete_link_uses_axis_aligned_pixels(self):
        values = [candidate(0, [0, 0]), candidate(1, [7, 7]), candidate(2, [14, 14])]
        self.assertEqual(arm2.complete_link_classes(values, 8), [[0, 1], [2]])

    def test_tie_uses_earliest_index(self):
        values = [candidate(0, [0, 0]), candidate(1, [100, 100])]
        self.assertEqual(arm2.select_mode(values, 7)["representative_index"], 0)

    def test_collision_uses_axis_aligned_pixels(self):
        self.assertTrue(arm2.collide([0, 0], [7, 7], 7))
        self.assertFalse(arm2.collide([0, 0], [8, 0], 7))

    def test_q1_summary_units(self):
        value = arm2.summarize_counts(Counter({"a": 2, "b": 1}))
        self.assertEqual(value["rows"], 3)
        self.assertEqual(value["screens"], 2)
        self.assertEqual(value["singleton_screen_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()