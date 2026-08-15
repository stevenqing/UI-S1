import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("q2", RUN_DIR / "q2.py")
q2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = q2
SPEC.loader.exec_module(q2)


def candidate(order, coordinate, action="CLICK"):
    return {"order": order, "coordinate": coordinate, "action": action, "parameter": "", "parse_ok": True}


class Q2Test(unittest.TestCase):
    def test_complete_link_does_not_bridge(self):
        values = [candidate(0, [0.0, 0.0]), candidate(1, [0.1, 0.0]), candidate(2, [0.2, 0.0])]
        self.assertEqual(q2.complete_link_classes(values, 0.11), [[0, 1], [2]])

    def test_mode_tie_uses_earliest_candidate(self):
        values = [candidate(0, [0.0, 0.0]), candidate(1, [1.0, 0.0])]
        self.assertEqual(q2.select_mode(values, 0.01)["representative_candidate_index"], 0)

    def test_collision_is_symmetric_and_same_screen(self):
        modes = {
            "a": {"image_sha256": "screen", "coordinate": [0.0, 0.0]},
            "b": {"image_sha256": "screen", "coordinate": [0.05, 0.0]},
            "c": {"image_sha256": "other", "coordinate": [0.0, 0.0]},
        }
        self.assertEqual(q2.collision_flags(modes, 0.1), {"a": True, "b": True, "c": False})

    def test_actions_do_not_block_cross_row_location_collision(self):
        modes = {
            "a": {"image_sha256": "screen", "coordinate": [0.0, 0.0], "action": "CLICK"},
            "b": {"image_sha256": "screen", "coordinate": [0.0, 0.0], "action": "TYPE"},
        }
        self.assertTrue(all(q2.collision_flags(modes, 0.0).values()))


if __name__ == "__main__":
    unittest.main()