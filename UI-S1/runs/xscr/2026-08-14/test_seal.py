import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_seal", RUN_DIR / "prepare_seal.py")
prepare_seal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_seal
SPEC.loader.exec_module(prepare_seal)


class SealTest(unittest.TestCase):
    def test_half_up_rounding(self):
        self.assertEqual(prepare_seal.selected_count(5, 0.3), 2)
        self.assertEqual(prepare_seal.selected_count(15, 0.3), 5)

    def test_exploratory_screen_is_retained(self):
        self.assertEqual(prepare_seal.selected_count(1, 0.3), 0)
        self.assertEqual(prepare_seal.selected_count(2, 0.9), 1)

    def test_assignment_is_order_invariant(self):
        rows = [
            {"image_sha256": "a", "fold": 0},
            {"image_sha256": "b", "fold": 0},
            {"image_sha256": "a", "fold": 0},
        ]
        left = prepare_seal.assign_screens(rows, "test", 7, 0.3)
        right = prepare_seal.assign_screens(list(reversed(rows)), "test", 7, 0.3)
        self.assertEqual(left, right)

    def test_cross_stratum_screen_is_rejected(self):
        rows = [
            {"image_sha256": "a", "fold": 0},
            {"image_sha256": "a", "fold": 1},
        ]
        with self.assertRaises(ValueError):
            prepare_seal.assign_screens(rows, "test", 7, 0.3)


if __name__ == "__main__":
    unittest.main()