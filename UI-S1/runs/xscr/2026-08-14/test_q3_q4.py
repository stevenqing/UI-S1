import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("q3_q4", RUN_DIR / "q3_q4.py")
q3_q4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = q3_q4
SPEC.loader.exec_module(q3_q4)


class BoundsTest(unittest.TestCase):
    def test_only_strictly_stronger_mode_displaces(self):
        row = {"sample_key": "a", "representative_coordinate": [0.0, 0.0], "mode_weight": 2}
        tied = {"sample_key": "b", "representative_coordinate": [0.0, 0.0], "mode_weight": 2}
        stronger = {"sample_key": "c", "representative_coordinate": [0.0, 0.0], "mode_weight": 3}
        self.assertFalse(q3_q4.loses_location(row, [row, tied], 0.0))
        self.assertTrue(q3_q4.loses_location(row, [row, stronger], 0.0))

    def test_summary_pairs_repair_and_damage(self):
        rows = [
            {"selected_correct": False, "recoverable": True, "repairable": True, "damageable": False, "target_coordinate": [0, 0], "multi_row_screen": True, "shared_target": False},
            {"selected_correct": True, "recoverable": False, "repairable": False, "damageable": True, "target_coordinate": [1, 1], "multi_row_screen": True, "shared_target": True},
        ]
        result = q3_q4.summarize(rows)
        self.assertEqual(result["signed_screening_proxy_pp"], 0.0)
        self.assertEqual(result["repairable_over_recoverable"], 1.0)
        self.assertEqual(result["damageable_over_selected_correct"], 1.0)


if __name__ == "__main__":
    unittest.main()